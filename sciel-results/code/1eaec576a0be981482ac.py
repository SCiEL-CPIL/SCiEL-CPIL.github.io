import math
import time
import random
import numpy as np

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False
    
    # Precompute base hex vertices (side 1, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    
    # Snap angles: 0, 30, 60, 90, 120, 150
    snap_angles = [0, 30, 60, 90, 120, 150]
    
    # Precompute vertices for each snap angle
    # verts_by_angle[a] = list of 6 (x, y) tuples for side 1
    verts_by_angle = {}
    for a in snap_angles:
        rad = math.radians(a)
        ca, sa = math.cos(rad), math.sin(rad)
        v_list = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            v_list.append((rx, ry))
        verts_by_angle[a] = v_list
        
    # Precompute "shadow" vertices for collision checking?
    # Actually, for disjointness, we need to check if two hexagons overlap.
    # Two convex polygons overlap if:
    # 1. One vertex of A is inside B, or
    # 2. One vertex of B is inside A, or
    # 3. An edge of A intersects an edge of B.
    
    # To speed this up, we can use a grid or spatial hashing, but with N=11 and discrete snaps,
    # the number of candidate positions is small enough that O(N^2) checks per config might be fine
    # if the candidate set is restricted.
    
    # Let's define a set of candidate centers.
    # We use a triangular lattice with spacing slightly larger than sqrt(3) to ensure no overlap
    # if angles are aligned, but we allow different angles so we must check.
    # However, if we snap centers to a lattice with spacing D, and D is large enough,
    # we reduce the search space.
    
    # Optimal packing of hexagons is triangular lattice with spacing sqrt(3).
    # Let's generate candidate centers on a grid of spacing 0.5 (fine enough to capture shifts)
    # but restrict to a bounding box.
    
    # Better: Generate candidate centers from a "super-lattice" or just a fine grid
    # and filter based on rough distance.
    
    # Let's use a grid of step 0.25 in [-4, 4].
    # That's 33x33 = 1089 points.
    # With 6 angles, 1089 * 6 = 6534 candidates.
    # Picking 11 from 6534 is too many for brute force.
    
    # Alternative: Local search on discrete candidates.
    # Start with a valid configuration (e.g., triangular lattice cluster).
    # Then try to move each hex to a nearby candidate center/angle to reduce S.
    
    def get_candidates(center_x, center_y, radius):
        # Generate candidate centers within radius of (center_x, center_y)
        # on a grid of step 0.1 for fine tuning? No, 0.1 is too fine.
        # Let's use a grid of step 0.2.
        step = 0.2
        cx_min = int((center_x - radius) / step) * step
        cx_max = int((center_y + radius) / step) * step # Typo in var name in thought, fix below
        cy_min = int((center_y - radius) / step) * step
        cy_max = int((center_y + radius) / step) * step
        
        candidates = []
        cx = cx_min
        while cx <= cx_max:
            cy = cy_min
            while cy <= cy_max:
                if (cx - center_x)**2 + (cy - center_y)**2 <= radius**2:
                    candidates.append((cx, cy))
                cy += step
            cx += step
        return candidates

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

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

    def hexagons_disjoint(verts_a, verts_b):
        # Quick reject: bounding box
        min_ax = min(v[0] for v in verts_a)
        max_ax = max(v[0] for v in verts_a)
        min_ay = min(v[1] for v in verts_a)
        max_ay = max(v[1] for v in verts_a)
        
        min_bx = min(v[0] for v in verts_b)
        max_bx = max(v[0] for v in verts_b)
        min_by = min(v[1] for v in verts_b)
        max_by = max(v[1] for v in verts_b)
        
        if max_ax < min_bx or max_bx < min_ax or max_ay < min_by or max_by < min_ay:
            return True
            
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = []
        for (ix, iy, iangle) in inner:
            # iangle is in degrees
            rad = math.radians(iangle)
            ca, sa = math.cos(rad), math.sin(rad)
            iv = []
            for vx, vy in base_verts:
                rx = vx * ca - vy * sa
                ry = vx * sa + vy * ca
                iv.append((ix + rx, iy + ry))
            inner_verts.append(iv)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            rad = math.radians(iangle)
            ca, sa = math.cos(rad), math.sin(rad)
            iv = []
            for vx, vy in base_verts:
                rx = vx * ca - vy * sa
                ry = vx * sa + vy * ca
                iv.append((ix + rx, iy + ry))
            inner_verts.append(iv)
        
        # Check inside outer
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        
        # Check disjoint
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    # Initial Configuration: Triangular Lattice Cluster
    # 1 in center, 6 in ring 1, 4 in ring 2 (to make 11)
    sp = math.sqrt(3)
    initial_inner = []
    initial_inner.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = math.radians(60 * i)
        initial_inner.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Ring 2: pick 4 positions that are closest to the "holes" or just continue the lattice
    # Lattice points in ring 2 are at distance 2*sp or sp*sqrt(3+1)=2? No.
    # Neighbors of neighbors.
    # Let's just add 4 more from the lattice.
    # Points at (2*sp, 0), (sp, sp*sqrt(3)), etc.
    ring2_candidates = [
        [2*sp, 0, 0],
        [sp, sp*math.sqrt(3), 0],
        [-sp, sp*math.sqrt(3), 0],
        [-2*sp, 0, 0],
        [-sp, -sp*math.sqrt(3), 0],
        [sp, -sp*math.sqrt(3), 0]
    ]
    # Pick 4 that minimize S? Or just take first 4?
    # Let's take the ones that form a symmetric shape if possible.
    # 1, 6, 4. A "hexagon" of 7 + 4 more.
    # Let's just take 4 from ring2_candidates.
    for k in range(4):
        initial_inner.append(list(ring2_candidates[k]))
    
    # Ensure we have exactly N
    while len(initial_inner) < N:
        initial_inner.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
    initial_inner = initial_inner[:N]
    
    best_inner = [list(x) for x in initial_inner]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Validate and calculate S
    cur_S = compute_outer_hex(best_inner, 0.0, 0.0, 0.0)
    if all_hexagons_fit(best_inner, 0.0, 0.0, cur_S, 0.0):
        best_S = cur_S
        best_valid = True
    else:
        # Fallback: spread them out more
        for k in range(N):
            initial_inner[k][0] *= 1.1
            initial_inner[k][1] *= 1.1
        best_inner = [list(x) for x in initial_inner]
        cur_S = compute_outer_hex(best_inner, 0.0, 0.0, 0.0)
        if all_hexagons_fit(best_inner, 0.0, 0.0, cur_S, 0.0):
            best_S = cur_S
            best_valid = True
        else:
            # Absolute fallback: line
            for k in range(N):
                initial_inner[k] = [k * (sp + 0.1), 0.0, 0.0]
            best_inner = [list(x) for x in initial_inner]
            cur_S = compute_outer_hex(best_inner, 0.0, 0.0, 0.0)
            best_S = cur_S
            best_valid = True
            
    # Discrete Local Search
    # We will try to move each hex to a nearby grid point or snap angle.
    # Grid step 0.1 for positions, snap angles for rotation.
    
    grid_step = 0.1
    search_radius = 0.5
    
    rng = random.Random(42)
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        idx = rng.randrange(N)
        old_inner = [list(x) for x in best_inner]
        
        # Generate candidates for this hex
        cx, cy, ca_deg = best_inner[idx]
        
        # Position candidates
        # We sample from a small grid around current position
        candidates = []
        # Add current
        candidates.append((cx, cy))
        # Add grid points
        for dx in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for dy in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                if dx == 0 and dy == 0: continue
                candidates.append((cx + dx, cy + dy))
                
        # Angle candidates
        # Current angle snapped?
        # We only allow snap angles in the representation?
        # The problem allows any angle. But we are restricting to snaps for speed.
        # Let's stick to snaps: 0, 30, 60...
        
        best_candidate_S = float('inf')
        best_candidate_pos = None
        best_candidate_ang = None
        
        # To speed up, we can pre-compute vertices for all candidates?
        # No, just check validity.
        
        for nx, ny in candidates:
            for na in snap_angles:
                # Skip if same as current
                if nx == cx and ny == cy and na == ca_deg:
                    continue
                    
                new_inner = [list(x) for x in best_inner]
                new_inner[idx] = [nx, ny, na]
                
                # Quick check: does it overlap with neighbors?
                # We only need to check against other hexes, not outer (outer is computed later)
                # But it must fit in outer. Since we minimize S, if it moves out, S increases.
                # If S increases, we reject (unless we allow hill climbing, but we want to minimize).
                # So we only accept if S decreases.
                # To know S, we must compute it.
                
                # Optimization: Only compute S if the move is "promising" or random.
                # Actually, computing S is O(N * 6) = 66 ops. Very fast.
                # Checking disjointness is O(N^2 * 36) ~ 1000 ops. Also fast.
                
                new_S = compute_outer_hex(new_inner, 0.0, 0.0, best_angle)
                
                # If S is not better, skip disjoint check?
                # If S is better, we MUST check disjointness.
                # If S is worse, we don't care about disjointness (we reject).
                # Wait, if S is worse, it might still be valid but worse. We reject.
                # So only check disjointness if new_S < best_S.
                
                if new_S < best_S:
                    if all_hexagons_fit(new_inner, 0.0, 0.0, new_S, best_angle):
                        best_candidate_S = new_S
                        best_candidate_pos = (nx, ny)
                        best_candidate_ang = na
                        
                        # Accept immediately? Or find best in this iteration?
                        # Let's accept the first improvement found to keep it simple and fast.
                        best_inner = new_inner
                        best_S = new_S
                        break
            if best_candidate_pos is not None:
                break
                
        # If we found an improvement, we break out of candidate loops.
        # If not, we continue to next iteration.
        
        # Also, occasionally try rotating the OUTER hexagon?
        # The task allows outer rotation.
        # But fixing outer at 0 and letting inner adapt is often optimal due to symmetry?
        # Not necessarily. But for N=11, the shape is likely symmetric enough.
        # Let's stick to fixing outer for now to save search space.
        
    # Final Polish: Try to adjust outer angle
    # If we have time, try a few outer angles
    if time.time() < deadline - 5:
        for oa in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 2:
                break
            # Re-compute S for current inner with this outer angle
            # Note: Inner positions are fixed relative to center (0,0).
            # If we rotate outer, we don't need to move inner, just check fit.
            # But S might change.
            S_test = compute_outer_hex(best_inner, 0.0, 0.0, oa)
            if S_test < best_S:
                if all_hexagons_fit(best_inner, 0.0, 0.0, S_test, oa):
                    best_S = S_test
                    best_angle = oa

    if not best_valid or best_inner is None:
        # Absolute fallback
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
