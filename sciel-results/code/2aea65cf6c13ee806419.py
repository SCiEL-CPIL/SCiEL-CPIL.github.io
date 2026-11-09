    import numpy as np
    from scipy.optimize import minimize, minimize_scalar

    N = 11

    # EVOLVE-BLOCK-START
    def _hex_vertices(angle_deg, cx, cy):
        """Vertices of a regular hexagon with side 1, centered at (cx,cy), rotated by angle_deg."""
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        # Base vertices at angle 0: (1,0), (0.5, sqrt(3)/2), ...
        # Actually, standard regular hexagon vertices with side 1 (radius 1) are at angles 0, 60, ...
        # Wait, side length 1 means radius 1.
        base_angles = np.deg2rad(np.arange(0, 360, 60))
        base = np.vstack([np.cos(base_angles), np.sin(base_angles)]).T
        R = np.array([[c, -s], [s, c]])
        return R @ base.T + np.array([cx, cy])[:, None]

    def _point_in_hex(pt, hex_verts):
        """Check if point is inside hexagon. Hex_verts is (6,2)."""
        # Use cross product orientation
        inside = True
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            # Cross product (p2-p1) x (pt-p1)
            v1 = p2 - p1
            v2 = pt - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            # Assuming CCW order, cross should be positive for points inside
            # Let's check orientation of hex_verts
            # If hex_verts are CCW, cross > 0 means left of edge, which is inside.
            if cross < -1e-9:
                return False
        return True

    def _min_dist_convex_poly(P, Q):
        """
        Minimum distance between two convex polygons P and Q (lists of (2,) arrays).
        Returns 0 if they intersect.
        """
        # 1. Check for intersection
        # If any vertex of P is in Q, return 0
        for p in P:
            if _point_in_hex(p, Q):
                return 0.0
        for q in Q:
            if _point_in_hex(q, P):
                return 0.0
        
        # 2. Calculate min distance between edges
        min_d = np.inf
        for i in range(6):
            p1 = P[i]
            p2 = P[(i + 1) % 6]
            for j in range(6):
                q1 = Q[j]
                q2 = Q[(j + 1) % 6]
                # Distance between segment p1-p2 and q1-q2
                # Approximation: min distance between any vertex of one and segment of other
                # This is O(12) per pair, total 6*6*12 = 432 ops per pair.
                # For 55 pairs, ~24000 ops. Acceptable.
                
                # Vectorized point-segment distance
                # dist(p, seg(a,b))
                
                # p1 to q1-q2
                d1 = _point_seg_dist(p1, q1, q2)
                # p2 to q1-q2
                d2 = _point_seg_dist(p2, q1, q2)
                # q1 to p1-p2
                d3 = _point_seg_dist(q1, p1, p2)
                # q2 to p1-p2
                d4 = _point_seg_dist(q2, p1, p2)
                
                d = min(d1, d2, d3, d4)
                if d < min_d:
                    min_d = d
        return min_d

    def _point_seg_dist(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-12:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def _min_outer_side(vertices, outer_angle_deg):
        """
        Calculate minimum side length of a regular hexagon (centered at origin, rotated by outer_angle_deg)
        that contains all points in `vertices` (shape (M, 2)).
        """
        # Normals to the sides of a regular hexagon of side S (apothem A = S*sqrt(3)/2)
        # The sides are at angles outer_angle_deg + 0, 60, ...?
        # Let's define the hexagon such that its vertices are at angles outer_angle_deg + 30 + 60k?
        # Or simpler: The support function h(n) = max(p . n).
        # For a regular hexagon centered at origin with side S, the width in direction n is 2 * A * |cos(angle_diff)|?
        # No. The apothem is A. The distance from center to side with normal n is A if n is aligned with a normal.
        # The condition for p to be inside is p . n_k <= A for all 6 normals n_k.
        # So A >= max_k (max_p p . n_k).
        # Also -A >= min_k (min_p p . n_k) => A >= -min_k (min_p p . n_k) = max_k (-min_p p . n_k).
        # So A >= max( max_k max_p p . n_k, max_k (-min_p p . n_k) ).
        # Let's compute max_p p . n_k for each k.
        
        a_outer = np.radians(outer_angle_deg)
        # Normals of the 6 sides.
        # If vertices are at angles 0, 60, ..., normals are at 30, 90, ...
        # Let's assume the "standard" hexagon has a flat top/bottom?
        # Vertices at 0, 60, 120, 180, 240, 300.
        # Normals at 30, 90, 150, 210, 270, 330.
        normal_angles = np.deg2rad(30 + 60 * np.arange(6) + outer_angle_deg)
        normals = np.vstack([np.cos(normal_angles), np.sin(normal_angles)]).T
        
        # Compute max dot product for each normal
        # dots = vertices @ normals.T  (M, 6)
        dots = vertices @ normals.T
        max_dots = np.max(dots, axis=0)
        min_dots = np.min(dots, axis=0)
        
        # A must be >= max(max_dots) and >= max(-min_dots)
        A = np.max(np.maximum(max_dots, -min_dots))
        
        S = 2.0 * A / np.sqrt(3.0)
        return S

    def construct():
        n = N
        rng = np.random.default_rng(42)
        
        # Initial guess: Hexagonal grid
        positions = []
        # Generate a hex grid
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = 1.5 * col + 0.75 * row
                y = np.sqrt(3) / 2 * row
                # Scale up slightly to avoid initial overlap
                x *= 1.1
                y *= 1.1
                positions.append([x, y, 0.0])
        
        positions = positions[:n]
        if len(positions) < n:
            # Pad if necessary
            while len(positions) < n:
                positions.append([rng.uniform(-5, 5), rng.uniform(-5, 5), 0.0])
        
        x0 = []
        for (x, y, a) in positions:
            x0.extend([x, y, np.radians(a)])
        x0 = np.array(x0, dtype=float)
        
        def objective(params):
            # Extract vertices
            verts_list = []
            all_verts = []
            for i in range(n):
                xi = params[3*i]
                yi = params[3*i+1]
                ai = params[3*i+2]
                verts = _hex_vertices(np.degrees(ai), xi, yi)
                verts_list.append(verts)
                all_verts.append(verts)
            
            all_verts = np.vstack(all_verts)
            
            # Calculate min outer side
            # Optimize outer angle in [0, 60]
            # We can use a simple grid or minimize_scalar
            # For speed, let's try a few angles
            best_S = np.inf
            for ang in np.linspace(0, 60, 13):
                S = _min_outer_side(all_verts, ang)
                if S < best_S:
                    best_S = S
            
            # Calculate min distance between inner hexes
            min_pair_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = _min_dist_convex_poly(verts_list[i], verts_list[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            
            # Penalty for overlap
            penalty = 0.0
            if min_pair_dist < 0:
                # This shouldn't happen with our distance function which returns 0 for intersection
                # But if it does, penalize
                penalty += 100.0 * abs(min_pair_dist)
            
            # Also penalize if min_pair_dist is very small?
            # No, we just want them to not overlap.
            # The objective is to minimize S.
            # If they overlap, the optimizer should push them apart.
            # To help the optimizer, we can add a small term to maximize min_pair_dist?
            # Or just rely on the fact that if they overlap, S might be smaller (if they clump),
            # but we want valid packings.
            # A better objective: S + lambda * max(0, -min_pair_dist + epsilon)
            # But _min_dist returns 0 for intersection.
            # We need a measure of "overlap depth".
            # Let's modify _min_dist_convex_poly to return negative distance if intersecting?
            # That's hard.
            # Alternative: Use a penalty based on the number of intersecting vertex-edge pairs?
            # Or simply: If min_pair_dist == 0, it's on the boundary.
            # We want min_pair_dist > 0.
            # Let's add a soft constraint: maximize min_pair_dist.
            # Objective: S - alpha * min_pair_dist?
            # If min_pair_dist is 0, we lose alpha * 0.
            # If we separate them, min_pair_dist increases, S might increase.
            # This is a trade-off.
            
            # Let's try: Objective = S + penalty
            # penalty = 0 if min_pair_dist > 0
            # penalty = 10 * (1 - min_pair_dist) if min_pair_dist <= 0? No.
            
            # Let's use a barrier function.
            # We want min_pair_dist > 0.
            # Let's define penalty = 100 * max(0, 0.1 - min_pair_dist)
            # This pushes them to be at least 0.1 apart? No, that changes the packing.
            # We want them to be as close as possible (touching).
            # So we want to minimize S, subject to min_pair_dist >= 0.
            
            # Since we can't easily handle hard constraints in this simple loop,
            # let's assume the optimizer will find a non-overlapping configuration
            # because overlapping configurations allow S to be smaller (clumping),
            # but the penalty should discourage it.
            
            # Let's refine the penalty:
            # If min_pair_dist is 0, they touch or overlap.
            # We need to distinguish touch vs overlap.
            # My _min_dist_convex_poly returns 0 for both.
            # I need to know if they are strictly inside.
            
            # Let's change _min_dist_convex_poly to return a negative value if overlapping.
            # If vertex of P is inside Q, return -distance_to_boundary?
            # This is complex.
            
            # Simpler: Use a large penalty if any vertex of P is inside Q or vice versa.
            # This is a boolean check.
            # Let's add a count of violations.
            violations = 0
            for i in range(n):
                for j in range(i+1, n):
                    # Check if any vertex of i is inside j
                    for v in verts_list[i]:
                        if _point_in_hex(v, verts_list[j]):
                            violations += 1
                    for v in verts_list[j]:
                        if _point_in_hex(v, verts_list[i]):
                            violations += 1
            
            penalty = 10.0 * violations
            
            return best_S + penalty

        bounds = [(-10, 10)] * (3 * n)
        for i in range(n):
            bounds[3*i+2] = (-np.pi, np.pi)
        
        best_result = None
        best_obj = np.inf
        
        for trial in range(5):
            if trial == 0:
                start = x0.copy()
            else:
                start = x0 + rng.normal(0, 1.0, size=3*n)
            
            try:
                res = minimize(objective, start, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_result = res.x
            except Exception as e:
                pass
        
        if best_result is None:
            best_result = x0
        
        inner = []
        for i in range(n):
            x = float(best_result[3*i])
            y = float(best_result[3*i+1])
            a_deg = float(np.degrees(best_result[3*i+2]))
            inner.append([x, y, a_deg])
        
        # Compute final outer side
        all_verts = []
        for (x, y, a_deg) in inner:
            verts = _hex_vertices(a_deg, x, y)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        best_S = np.inf
        for ang in np.linspace(0, 60, 25):
            S = _min_outer_side(all_verts, ang)
            if S < best_S:
                best_S = S
        
        return inner, (0.0, 0.0), float(best_S), 0.0
    # EVOLVE-BLOCK-END
    