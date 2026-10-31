import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def point_in_polygon(p, polygon):
        """Ray casting point in polygon test."""
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
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
           
        # Handle collinear/endpoint cases
        if d1 == 0 and point_on_segment(p3, p4, p1): return True
        if d2 == 0 and point_on_segment(p3, p4, p2): return True
        if d3 == 0 and point_on_segment(p1, p2, p3): return True
        if d4 == 0 and point_on_segment(p1, p2, p4): return True
        return False

    def point_on_segment(a, b, p):
        """Check if point p is on segment a-b."""
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) > 1e-9:
            return False
        dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
        length_sq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        return dot >= -1e-9 and dot <= length_sq + 1e-9

    def hexagons_disjoint(verts_a, verts_b):
        """Check if two hexagons (given as vertex lists) are disjoint."""
        # Check vertices of A inside B
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        # Check vertices of B inside A
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        # Check edge intersections
        for i in range(6):
            if segments_intersect(verts_a[i], verts_a[(i+1) % 6], verts_b[i], verts_b[(i+1) % 6]):
                return False
            if segments_intersect(verts_a[i], verts_a[(i+1) % 6], verts_b[(i+1) % 6], verts_b[i]):
                return False
        return True

    def compute_required_S(inner, ocx, ocy, oangle):
        """
        Compute the minimum side length S of the outer hexagon centered at (ocx, ocy) 
        with angle oangle such that it contains all inner hexagons.
        """
        # The outer hexagon is defined by 6 half-planes.
        # For a hexagon of side S, apothem a = S * sqrt(3) / 2.
        # The constraint is that for each of the 6 normals n_i, 
        # max dot(p - center, n_i) <= a
        # So S >= max(dot) / (sqrt(3)/2)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        
        # Generate vertices for all inner hexagons
        all_verts = []
        for (ix, iy, iangle) in inner:
            iv = hex_vertices(ix, iy, 1.0, iangle)
            all_verts.extend(iv)
        
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for v in all_verts:
                proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
            # Need S * apothem_factor >= max_proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    # --- Search Strategy: Backtracking with Forward Checking ---
    
    def backtrack_search(placed_hexes, S_limit, oangle, depth, rng):
        """
        Try to place (N - depth) more hexagons given 'placed_hexes' (list of (x,y,ang)).
        Returns a list of remaining hexagons if successful, else None.
        """
        if time.time() > deadline:
            return None
            
        if depth == N:
            return []
            
        # Generate candidate positions for the next hexagon
        # We use a grid of potential positions relative to existing hexagons
        # To make it efficient, we only consider positions that are "close" to existing ones
        # or within a bounding box defined by the current S_limit.
        
        # Bounding box for the new hexagon center
        # The new hexagon must fit inside the outer hexagon of side S_limit.
        # Approximate bounding box:
        # The outer hexagon apothem is S_limit * sqrt(3)/2.
        # The inner hexagon apothem is 1 * sqrt(3)/2.
        # So the center must be within distance (S_limit - 1) * sqrt(3)/2 from the center?
        # Actually, simpler: just iterate over a grid of candidate positions.
        
        # To speed up, we define a set of "snap points" or a fine grid.
        # Let's try a grid approach with step size 0.25.
        # Range: +/- S_limit * 1.5
        step = 0.25
        range_val = S_limit * 1.5 + 2.0
        
        # Optimization: Instead of scanning the whole grid, scan near existing hexagons first?
        # For simplicity and robustness in this time limit, we scan a grid but limit the range.
        
        candidates = []
        
        # Heuristic: Start with positions close to existing hexagons to maintain cluster compactness
        # Add random positions to escape local minima
        existing_centers = [(x, y) for (x, y, _) in placed_hexes]
        
        # 1. Grid points near existing hexagons
        for (ex, ey) in existing_centers:
            for dx in np.arange(-2.5, 2.51, step):
                for dy in np.arange(-2.5, 2.51, step):
                    candidates.append((ex + dx, ey + dy))
        
        # 2. Random points in the bounding box
        for _ in range(50):
            rx = rng.uniform(-range_val, range_val)
            ry = rng.uniform(-range_val, range_val)
            candidates.append((rx, ry))
            
        # Shuffle to ensure randomness
        rng.shuffle(candidates)
        
        # Try placing the next hexagon at each candidate position
        # We restrict angles to 0, 30, 60, 90, 120, 150 for efficiency, 
        # but allow continuous rotation? The problem says "may rotate".
        # Continuous rotation makes the search space huge. 
        # However, for hexagons, optimal packings often align. 
        # Let's try a discrete set of angles first: 0, 30, 60, 90, 120, 150.
        # If that fails, we might need finer angles, but let's stick to 6 for speed.
        
        angles_to_try = [0, 30, 60, 90, 120, 150]
        
        for (cx, cy) in candidates:
            if time.time() > deadline:
                return None
            
            for ang in angles_to_try:
                # Check if this candidate fits with existing ones
                new_hex_verts = hex_vertices(cx, cy, 1.0, ang)
                
                valid = True
                for (px, py, pang) in placed_hexes:
                    existing_verts = hex_vertices(px, py, 1.0, pang)
                    if not hexagons_disjoint(new_hex_verts, existing_verts):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check if it fits in the outer hexagon of side S_limit
                # We don't need to check the full outer hexagon yet, 
                # but we can check if the center is too far out.
                # A quick check: distance from center.
                # Max distance from center to a point in inner hex is 2.0.
                # Max distance from center to boundary of outer hex is S_limit.
                # So if sqrt(cx^2 + cy^2) + 2.0 > S_limit, it definitely doesn't fit.
                if math.sqrt(cx*cx + cy*cy) + 2.0 > S_limit + 1e-9:
                    continue
                
                # If it passes these checks, recurse
                new_placed = placed_hexes + [(cx, cy, ang)]
                result = backtrack_search(new_placed, S_limit, oangle, depth + 1, rng)
                if result is not None:
                    return [(cx, cy, ang)] + result
                    
        return None

    def try_pack(S_limit, oangle, seed):
        """
        Attempt to pack N hexagons into an outer hexagon of side S_limit.
        Returns the list of inner hexagons if successful.
        """
        rng = random.Random(seed)
        # Start with an empty list, or pre-place one in the center to break symmetry?
        # Placing one in the center (0,0) is a good heuristic.
        initial = [(0.0, 0.0, 0.0)]
        
        # Check if initial fits
        init_verts = hex_vertices(0.0, 0.0, 1.0, 0.0)
        # It always fits in S_limit >= 1.0
        
        result = backtrack_search(initial, S_limit, oangle, 1, rng)
        if result is not None:
            return initial + result
        return None

    # --- Main Loop ---
    
    # We want to minimize S.
    # We can use a binary search on S, or simply try a sequence of S values.
    # Given the discrete nature of the search, binary search is tricky if the search is stochastic.
    # Let's try a "descent" approach: start with a large S, find a packing, then try to reduce S.
    
    # Initial guess for S. 
    # Area of 11 hexagons = 11 * (3*sqrt(3)/2) * 1^2 ≈ 11 * 2.598 ≈ 28.58
    # Area of outer hexagon = (3*sqrt(3)/2) * S^2
    # S^2 ≈ 28.58 / 2.598 ≈ 11.0
    # S ≈ sqrt(11) ≈ 3.32
    # However, packing efficiency is < 100%. 
    # Best known packing density for hexagons in a hexagon is high, but for small N, it's lower.
    # Let's start with S = 4.5 and try to go down.
    
    current_S = 4.5
    oangle = 0.0
    
    # To improve the search, we can also vary the outer angle.
    # But let's stick to 0 for now, as the inner hexagons can rotate.
    
    packings_found = 0
    
    while time.time() < deadline and current_S > 3.0:
        # Try to find a packing for the current S
        # Use multiple random seeds to increase chances
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            inner = try_pack(current_S, oangle, seed)
            if inner is not None:
                found = True
                # Verify the packing is valid and compute the exact minimal S for this configuration
                # The S_limit used in search might be larger than the minimal required S.
                exact_S = compute_required_S(inner, 0.0, 0.0, oangle)
                
                if exact_S < best_S:
                    best_S = exact_S
                    best_inner = inner
                    best_center = (0.0, 0.0)
                    best_angle = oangle
                    
                # If we found a packing, we can try to reduce S
                # The reduction step depends on how much slack we have.
                # If exact_S is significantly smaller than current_S, we can jump down.
                # Otherwise, take a small step.
                if exact_S < current_S * 0.95:
                    current_S = exact_S * 1.05 # Try slightly larger than the found minimal
                else:
                    current_S -= 0.05
                break
        
        if not found:
            # If no packing found for this S, increase S a bit to ensure we can find one
            current_S += 0.1
            # If we keep failing, maybe the angle is bad? Try changing angle?
            # For now, just increase S.
            
    # If we didn't find anything (unlikely), fallback
    if best_inner is None:
        # Fallback: simple grid
        sp = math.sqrt(3)
        inner = []
        for i in range(N):
            # Arrange in a line or simple cluster
            x = (i % 3 - 1) * sp
            y = (i // 3 - 1) * sp * 1.5
            inner.append([x, y, 0.0])
        best_S = compute_required_S(inner, 0.0, 0.0, 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
