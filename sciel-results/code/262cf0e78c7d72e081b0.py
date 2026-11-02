"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 150s limit, 5s safety margin
    
    # 1. Initialize positions in a dense hexagonal lattice
    # Hexagonal lattice vectors for side length 1 (flat-top hexagons)
    # Distance between centers of adjacent hexagons in a perfect lattice:
    # For flat-top, x-spacing is 1.5 * side, y-spacing is sqrt(3) * side
    # Wait, standard hex packing of unit hexagons:
    # If side length is 1, the width (pointy-to-pointy) is 2, height (flat-to-flat) is sqrt(3).
    # To pack them disjointly in a lattice:
    # x_step = 1.5
    # y_step = sqrt(3)
    # Even columns shifted by sqrt(3)/2
    
    x_step = 1.5
    y_step = np.sqrt(3)
    
    # Create a grid of potential positions
    # We want N hexagons. Let's arrange them in a roughly hexagonal shape or just a grid.
    # A grid of width ~sqrt(N) and height ~sqrt(N) is a good start.
    side_grid = int(np.ceil(np.sqrt(N)))
    
    positions = []
    angles = [0.0] * N
    
    # Generate lattice points
    count = 0
    # Iterate over a sufficiently large grid and pick the first N
    # To make it compact, we should pick points closest to the origin.
    # Let's generate a 2D array of lattice points and sort by distance to origin.
    
    max_range = 10 # Sufficient for N=11
    candidates = []
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            # Hex lattice coordinates to Cartesian
            x = x_step * i
            y = y_step * (j + (i + abs(i)) / 2.0)
            candidates.append((x, y))
            
    # Sort by distance to origin to get the most central, dense cluster
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N
    initial_pos = candidates[:N]
    
    # 2. Relaxation Algorithm
    # We will simulate "soft" collisions.
    # We define a function to check if two hexagons overlap.
    # Since they are regular hexagons, we can check vertex containment or edge intersection.
    # However, for speed, we can use a conservative bounding circle or exact polygon intersection.
    # Given N=11, exact polygon intersection is fast enough.
    
    def hex_vertices(center, angle_deg):
        cx, cy = center
        theta = np.deg2rad(angle_deg)
        verts = []
        for k in range(6):
            phi = theta + np.deg2rad(60 * k)
            verts.append((cx + np.cos(phi), cy + np.sin(phi)))
        return np.array(verts)

    def point_in_poly(pt, poly):
        # Ray casting or cross product method
        # Using cross product for convex poly
        n = len(poly)
        sign = 0
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            # Vector p1->p2 and p1->pt
            v1 = p2 - p1
            v2 = pt - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross > 1e-9:
                if sign < 0: return False
                sign = 1
            elif cross < -1e-9:
                if sign > 0: return False
                sign = -1
        return True

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def hexagons_intersect(v1, v2):
        # Check if any vertex of v1 is inside v2
        for p in v1:
            if point_in_poly(p, v2):
                return True
        for p in v2:
            if point_in_poly(p, v1):
                return True
        # Check edge intersections
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def check_all_disjoint(pos, angles):
        verts = [hex_vertices(pos[i], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_intersect(verts[i], verts[j]):
                    return False
        return True

    def get_min_container(pos, angles):
        # Find the minimum side length of a regular hexagon (rotated to match inner? No, outer can rotate)
        # The task says outer hexagon center (x,y), side length and rotation.
        # To minimize side length, we should align the outer hexagon with the "tightest" direction of the inner cluster.
        # A simple heuristic: compute the bounding box of the union of all inner hexagons in several rotations of the outer frame?
        # Actually, the outer hexagon is fixed orientation relative to its own coordinate system, but we can rotate the whole assembly?
        # No, "outer hexagon center (x, y), side length and rotation". This implies the outer hexagon has a specific orientation.
        # The score is the outer side length.
        # We can choose the outer rotation to minimize the required side length.
        # Let's try 6 rotations (0, 30, 60, 90, 120, 150 degrees) of the outer hexagon relative to the inner cluster's principal axes?
        # Or simpler: The optimal outer hexagon for a set of points is related to the width of the set in 3 directions.
        # For a regular hexagon container, the condition for a point (x,y) to be inside a hexagon of side S centered at (0,0) with flat-top orientation (vertices at 0, 60, ...):
        # The hexagon is the intersection of 3 strips.
        # Strip 1 (horizontal): |y| <= S * sqrt(3) / 2
        # Strip 2 (60 deg): |x * sin(60) + y * cos(60)| <= S * sqrt(3) / 2  => |x*sqrt(3)/2 + y/2| <= S*sqrt(3)/2
        # Strip 3 (120 deg): |x * sin(120) + y * cos(120)| <= S * sqrt(3) / 2 => |x*sqrt(3)/2 - y/2| <= S*sqrt(3)/2
        
        # So for a given orientation of the outer hexagon (angle alpha), we can transform the points (x,y) into the outer hexagon's coordinate system.
        # Or, we can just compute the required S for a given orientation.
        # Let's define the outer hexagon vertices for side S and angle alpha.
        # It's easier to check containment:
        # A point (px, py) is inside the outer hexagon (center 0, side S, angle alpha) if:
        # The distance from the point to each of the 3 axes of symmetry is <= S * sqrt(3) / 2.
        # The axes of symmetry for a flat-top hexagon at angle 0 are at 90, 210, 330 degrees? No.
        # Flat-top hexagon: Vertices at 0, 60, 120, 180, 240, 300.
        # Normals to sides are at 30, 90, 150, 210, 270, 330.
        # The 3 pairs of parallel sides correspond to normals at 30, 90, 150 (and opposite).
        # Distance from origin to side is S * sqrt(3) / 2.
        # So, for a point (x,y), the "radius" in direction theta is r(theta).
        # The hexagon contains the point if max over the 3 normal directions of (x*n_x + y*n_y) <= S*sqrt(3)/2.
        
        # We want to minimize S.
        # S = max( max_i (x_i * n_kx + y_i * n_ky) ) * 2 / sqrt(3) for the 3 normals k.
        # We can rotate the outer hexagon by angle alpha. The normals rotate by alpha.
        # We can sample a few alphas to find the minimum S.
        
        # Let's compute the "width" of the set of vertices of all inner hexagons.
        all_verts = []
        for i in range(N):
            all_verts.extend(hex_vertices(pos[i], angles[i]).tolist())
        all_verts = np.array(all_verts)
        
        min_S = float('inf')
        # Try 12 rotations of the outer hexagon (every 15 degrees) to be safe
        for alpha_deg in np.arange(0, 180, 15):
            alpha = np.deg2rad(alpha_deg)
            # Normals for flat-top hexagon rotated by alpha
            # Base normals (flat-top, side S):
            # n1 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
            # n2 = (cos(90), sin(90)) = (0, 1)
            # n3 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
            # Rotate these by alpha
            
            base_normals = [
                np.array([np.cos(np.pi/6), np.sin(np.pi/6)]),
                np.array([0, 1]),
                np.array([np.cos(5*np.pi/6), np.sin(5*np.pi/6)])
            ]
            
            max_proj = 0
            for n in base_normals:
                # Rotate normal
                n_rot = np.array([
                    n[0]*np.cos(alpha) - n[1]*np.sin(alpha),
                    n[0]*np.sin(alpha) + n[1]*np.cos(alpha)
                ])
                # Project all vertices
                projs = all_verts @ n_rot
                max_proj = max(max_proj, np.max(projs))
                
            S = max_proj * 2 / np.sqrt(3)
            if S < min_S:
                min_S = S
                best_alpha = alpha_deg
                
        return min_S, best_alpha

    # Relaxation Loop
    pos = [np.array(p) for p in initial_pos]
    current_S, current_alpha = get_min_container(pos, angles)
    best_S = current_S
    best_pos = [p.copy() for p in pos]
    best_alpha = current_alpha
    
    # We will push apart overlapping hexagons.
    # Step size decay
    step = 0.1
    temp = 1.0
    
    for iter in range(500):
        if time.time() > deadline:
            break
            
        # Check collisions
        verts = [hex_vertices(pos[i], angles[i]) for i in range(N)]
        collisions = []
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_intersect(verts[i], verts[j]):
                    collisions.append((i, j))
                    
        if not collisions:
            # No collisions, try to shrink container?
            # The container size is determined by the positions.
            # We can try to "pull" them towards the center to reduce the bounding hexagon.
            # Or just accept this configuration.
            # Let's do a small random jitter to escape local minima?
            # No, if no collisions, the current S is valid.
            # We can try to optimize S by moving points.
            # Simple gradient: move each point slightly towards the centroid to reduce spread.
            centroid = np.mean(pos, axis=0)
            for i in range(N):
                pos[i] = pos[i] + 0.01 * (centroid - pos[i])
            new_S, new_alpha = get_min_container(pos, angles)
            if new_S < best_S:
                best_S = new_S
                best_pos = [p.copy() for p in pos]
                best_alpha = new_alpha
            # If we get stuck (S doesn't improve), we might be in a local min.
            # Add small noise
            if iter % 10 == 0:
                for i in range(N):
                    pos[i] += np.random.normal(0, 0.01, 2)
            continue
            
        # Resolve collisions
        for i, j in collisions:
            # Push apart
            center_i = pos[i]
            center_j = pos[j]
            diff = center_j - center_i
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.array([np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
                dist = np.linalg.norm(diff)
            dir = diff / dist
            # Push each by half the step
            pos[i] -= dir * step
            pos[j] += dir * step
            
        # Decay step
        step *= 0.95
        temp *= 0.99
        
        # Update best
        # Only update best if configuration is valid (no collisions)
        # Since we just pushed apart, they might still overlap if step was too small or multiple collisions.
        # So we check validity before updating best.
        verts = [hex_vertices(pos[i], angles[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_intersect(verts[i], verts[j]):
                    valid = False
                    break
            if not valid: break
            
        if valid:
            new_S, new_alpha = get_min_container(pos, angles)
            if new_S < best_S:
                best_S = new_S
                best_pos = [p.copy() for p in pos]
                best_alpha = new_alpha
                
    # Final polish: high precision check and slight optimization
    # Use best_pos
    pos = best_pos
    
    # One more relaxation with very small steps to ensure disjointness and minimal S
    for iter in range(100):
        if time.time() > deadline - 2:
            break
        verts = [hex_vertices(pos[i], angles[i]) for i in range(N)]
        collisions = []
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_intersect(verts[i], verts[j]):
                    collisions.append((i, j))
        if not collisions:
            # Try to tighten
            centroid = np.mean(pos, axis=0)
            for i in range(N):
                pos[i] = pos[i] + 0.001 * (centroid - pos[i])
            new_S, new_alpha = get_min_container(pos, angles)
            if new_S < best_S:
                best_S = new_S
                best_pos = [p.copy() for p in pos]
                best_alpha = new_alpha
        else:
            for i, j in collisions:
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.array([0.001, 0])
                    dist = 0.001
                dir = diff / dist
                pos[i] -= dir * 0.01
                pos[j] += dir * 0.01

    # Final validation
    verts = [hex_vertices(best_pos[i], angles[i]) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagons_intersect(verts[i], verts[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to a known valid configuration if relaxation failed
        # Use the initial lattice if it was valid, or a very large one
        # The initial lattice with spacing 1.5, sqrt(3) is valid.
        # Let's re-generate initial positions and check.
        pos = [np.array(p) for p in initial_pos]
        verts = [hex_vertices(pos[i], angles[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_intersect(verts[i], verts[j]):
                    valid = False
                    break
        if not valid:
            # Extremely safe fallback: linear array with large spacing
            pos = [np.array([10.0 * k, 0.0]) for k in range(N)]
            
    final_S, final_alpha = get_min_container(pos, angles)
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angles[i]])
        
    return inner, (0.0, 0.0), final_S, final_alpha
# EVOLVE-BLOCK-END
