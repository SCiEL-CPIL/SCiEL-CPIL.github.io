import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot):
        angles = np.deg2rad(np.arange(6) * 60 + rot)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def point_in_hex(pt, center, s, rot):
        """Check if point is inside regular hexagon centered at 'center' with side s and rotation rot."""
        # Transform point to local coordinates of hexagon
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        # Rotate point by -rot
        rad = -np.deg2rad(rot)
        cos_r = np.cos(rad)
        sin_r = np.sin(rad)
        x_local = dx * cos_r - dy * sin_r
        y_local = dx * sin_r + dy * cos_r
        
        # Hexagon with vertices at angles 0, 60, ... has edges at 30, 90, ...
        # Distance to center must be <= s * cos(angle_diff) for the closest edge normal?
        # Actually, simpler: check against the 6 half-planes.
        # Normals are at angles 30 + k*60 (relative to hex orientation).
        # In local coords (rot=0), vertices are at 0, 60, 120...
        # Edges are perpendicular to directions 30, 90, 150...
        # The distance from center to edge is r = s * sqrt(3)/2.
        # For a point (x,y) in local coords, max over k of (x*cos(n_k) + y*sin(n_k)) <= r
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        dists = np.dot(n_vecs, [x_local, y_local])
        r = s * np.sqrt(3) / 2.0
        return np.all(dists <= r + 1e-9)

    def min_dist_hexes(v1, v2):
        """
        Calculate minimum distance between two convex hexagons.
        Returns 0 if they intersect.
        """
        # Fast reject: bounding box
        min_x1, max_x1 = np.min(v1[:,0]), np.max(v1[:,0])
        min_y1, max_y1 = np.min(v1[:,1]), np.max(v1[:,1])
        min_x2, max_x2 = np.min(v2[:,0]), np.max(v2[:,0])
        min_y2, max_y2 = np.min(v2[:,1]), np.max(v2[:,1])
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            # Bounding boxes don't overlap, min dist is distance between boxes
            dx = max(0, max_x1 - min_x2, max_x2 - min_x1)
            dy = max(0, max_y1 - min_y2, max_y2 - min_y1)
            return np.sqrt(dx*dx + dy*dy)

        # Check intersection
        def point_in_poly(pt, poly):
            x, y = pt
            n = len(poly)
            inside = False
            p1x, p1y = poly[0]
            for i in range(n):
                p2x, p2y = poly[(i+1)%n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if xinters < x:
                                    inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        # Vertex in poly check
        for i in range(6):
            if point_in_poly(v1[i], v2):
                return 0.0
        for i in range(6):
            if point_in_poly(v2[i], v1):
                return 0.0

        # Edge intersection check
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return 0.0

        # Calculate min distance
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            v = p2 - p1
            c2 = np.dot(v, v)
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                
                # Point p3 to line p1-p2
                w = p3 - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(p3 - p1)
                elif c1 >= c2:
                    d = np.linalg.norm(p3 - p2)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d: min_d = d
                
                # Point p4 to line p1-p2
                w = p4 - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(p4 - p1)
                elif c1 >= c2:
                    d = np.linalg.norm(p4 - p2)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d: min_d = d

        return min_d

    def is_valid(inner_params, outer_s, outer_rot):
        """
        inner_params: list of (x, y, angle)
        Check if all inner hexagons are disjoint and inside outer hexagon.
        Returns True if valid.
        """
        outer_center = np.array([0.0, 0.0])
        
        # Check containment
        # Outer hexagon vertices
        outer_verts = hex_verts(0.0, 0.0, outer_s, outer_rot)
        
        inner_verts_list = []
        for i in range(N):
            x, y, a = inner_params[i]
            v = hex_verts(x, y, 1.0, a)
            inner_verts_list.append(v)
            
            # Check if all vertices of inner hex are inside outer hex
            # Using half-plane test for outer hex
            # Outer hex normals
            normals = np.deg2rad(30 + np.arange(6)*60 + outer_rot)
            n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
            r = outer_s * np.sqrt(3) / 2.0
            
            for pt in v:
                dists = np.dot(n_vecs, pt)
                if np.any(dists > r + 1e-5):
                    return False

        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_hexes(inner_verts_list[i], inner_verts_list[j])
                if d < 1e-5:
                    return False

        return True

    # Strategy:
    # 1. Initialize a reasonable packing.
    # 2. Iteratively try to reduce outer_s.
    # 3. If invalid, try to "fix" the packing by moving hexagons apart (local search).
    
    # Initial guess: Hexagonal lattice
    # Side length of inner hex is 1. Distance between centers in lattice is 2*sqrt(3)/2 * 2? No.
    # Distance between centers of adjacent hexagons in a tight packing is 2 * (apothem) = sqrt(3).
    # Wait, apothem is sqrt(3)/2. Distance between centers is 2 * apothem = sqrt(3).
    # Let's start with a grid of hexagons.
    
    # We want to minimize outer_s.
    # A rough estimate: 12 hexagons. Area = 12 * (3*sqrt(3)/2) * 1^2 = 6*sqrt(3) ~ 10.39.
    # Outer hex area = (3*sqrt(3)/2) * S^2.
    # S^2 ~ 10.39 / 2.598 ~ 4. S ~ 2.
    # But packing efficiency is < 1. Let's start with S = 4.0.
    
    current_outer_s = 4.0
    current_outer_rot = 0.0
    
    # Initialize positions in a hexagonal pattern
    # Center at (0,0)
    # Neighbors at distance sqrt(3)
    # 1 center, 6 neighbors, 5 more?
    # Let's just place them in a circle first to ensure they are disjoint, then tighten.
    
    def get_initial_packing(S):
        # Place 12 hexagons in a ring around center? Or grid?
        # Grid is better for density.
        # Let's try a simple spiral or grid.
        # Grid with spacing 2.0 (safe)
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for i in range(6):
            angle = i * 60
            r = 2.0 # Safe distance
            x = r * np.cos(np.deg2rad(angle))
            y = r * np.sin(np.deg2rad(angle))
            positions.append((x, y, 0.0))
        # Ring 2: 5 more
        for i in range(5):
            angle = i * 72 + 30
            r = 3.8
            x = r * np.cos(np.deg2rad(angle))
            y = r * np.sin(np.deg2rad(angle))
            positions.append((x, y, 0.0))
        return positions

    best_inner = get_initial_packing(current_outer_s)
    best_outer_s = current_outer_s
    best_outer_rot = current_outer_rot
    
    # Verify initial
    if is_valid(best_inner, best_outer_s, best_outer_rot):
        pass
    else:
        # If invalid, increase S
        while not is_valid(best_inner, best_outer_s, best_outer_rot):
            best_outer_s *= 1.1
            if best_outer_s > 10.0: break

    # Local Search / Optimization
    # Objective: Minimize outer_s
    # Constraint: is_valid
    
    # We will use a simple iterative improvement:
    # 1. Try to shrink outer_s by a small amount.
    # 2. If invalid, identify "problem" hexagons (those outside or overlapping).
    # 3. Move them towards center or away from neighbors.
    
    # To make this faster, we can precompute vertices for current positions.
    
    def compute_vertices(inner_params):
        return [hex_verts(p[0], p[1], 1.0, p[2]) for p in inner_params]

    # Optimization Loop
    max_iter = 2000
    shrink_step = 0.01
    move_step = 0.05
    
    for iter in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Try to shrink
        new_s = best_outer_s - shrink_step
        if new_s < 2.0:
            new_s = 2.0
            
        # Check if current packing is valid with new_s
        # We need to re-check containment with new_s
        # Overlaps are independent of outer_s
        # So we can cache overlap status? 
        # Overlaps depend on positions only.
        # Containment depends on positions and outer_s.
        
        # Let's just check validity with new_s
        if is_valid(best_inner, new_s, best_outer_rot):
            best_outer_s = new_s
            continue # Success, try to shrink more
        
        # 2. If invalid, we need to adjust positions.
        # Find which hexagons are problematic.
        # Problem 1: Outside outer hex
        # Problem 2: Overlapping
        
        # To save time, let's do a random perturbation of all positions slightly
        # and see if it helps.
        
        # Better: Identify specific issues.
        # This is complex. Let's do a simple "jitter" and "centering" force.
        
        new_inner = [p[:] for p in best_inner]
        
        # Apply small random jitter
        for i in range(N):
            new_inner[i][0] += np.random.normal(0, move_step)
            new_inner[i][1] += np.random.normal(0, move_step)
            # Keep rotation 0 for simplicity? Or randomize?
            # Rotating might help fit better.
            # Let's fix rotation to 0 for now to reduce complexity, 
            # unless we find a better packing with rotation.
            # Actually, let's allow small rotation changes.
            new_inner[i][2] += np.random.normal(0, 5)
            
        # Check validity with current best_outer_s (not the shrunk one yet)
        # If we can make it valid at current S, we might be able to shrink later.
        # But we want to minimize S.
        # So, if new_s failed, we should try to fix the packing to fit in new_s.
        # Let's check if new_inner is valid in new_s.
        
        if is_valid(new_inner, new_s, best_outer_rot):
            best_inner = new_inner
            best_outer_s = new_s
            continue
            
        # If new_s failed even with jitter, maybe we should try to "pull" hexagons towards center
        # to reduce the required outer size.
        
        # Calculate center of mass
        com_x = np.mean([p[0] for p in best_inner])
        com_y = np.mean([p[1] for p in best_inner])
        
        # Shift towards origin
        shift_x = -com_x * 0.1
        shift_y = -com_y * 0.1
        
        new_inner2 = [p[:] for p in best_inner]
        for i in range(N):
            new_inner2[i][0] += shift_x
            new_inner2[i][1] += shift_y
            
        if is_valid(new_inner2, new_s, best_outer_rot):
            best_inner = new_inner2
            best_outer_s = new_s
            continue
            
        # If nothing worked, try a larger shrink step reduction? Or just accept current best.
        # To avoid getting stuck, we can occasionally increase S back slightly to escape local minima
        # but we want to minimize S.
        
        # Let's try to optimize the rotation of the outer hexagon?
        # Maybe rotating the outer hex helps.
        if np.random.rand() < 0.1:
            new_rot = best_outer_rot + np.random.normal(0, 5)
            if is_valid(best_inner, best_outer_s, new_rot):
                # If valid at same S, keep it. Maybe allows shrinking later?
                # Doesn't change S directly, but might allow smaller S.
                best_outer_rot = new_rot
                # Try to shrink again
                new_s2 = best_outer_s - shrink_step
                if new_s2 > 2.0 and is_valid(best_inner, new_s2, new_rot):
                    best_outer_s = new_s2
                    continue

    # Final check and return
    # Ensure we return a valid solution
    if not is_valid(best_inner, best_outer_s, best_outer_rot):
        # Fallback: Increase S until valid
        while not is_valid(best_inner, best_outer_s, best_outer_rot):
            best_outer_s *= 1.05
            if best_outer_s > 10.0:
                # Last resort: spread them out more
                # This shouldn't happen if initial was valid and we only shrink when valid
                # But if jitter moved them out, we might be invalid at current S.
                # So we need to find a valid config.
                # Reset to a known valid config?
                # Let's just return a very large valid one.
                break

    # Format output
    inner = []
    for x, y, a in best_inner:
        inner.append([x, y, a])
    
    return inner, (0.0, 0.0), best_outer_s, best_outer_rot

# Run a quick self-test to ensure no immediate crashes
if __name__ == "__main__":
    # This is just for local testing, not part of the submission
    pass
