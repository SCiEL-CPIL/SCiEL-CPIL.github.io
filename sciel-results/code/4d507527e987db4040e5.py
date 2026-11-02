<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse grid to find promising configurations
    for combo in all_combos:
        if time.time() > deadline - 30.0:  # Leave 30s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations
        for oang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    # Phase 2: Fine-tune using smooth soft-max surrogate for gradient-based optimization
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        def soft_max_outer_side(inner_hexes, cx, cy, angle_deg, beta=50.0):
            """Smooth approximation of _min_outer_side using log-sum-exp.
            Returns a smooth value close to the true min outer side."""
            a = -math.radians(angle_deg)
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            
            # Collect all projection values
            projs = []
            for cxh, cyh, s, ang in inner_hexes:
                verts = _hex_vertices(cxh, cyh, s, ang)
                for vx, vy in verts:
                    dx = vx - cx
                    dy = vy - cy
                    lx = dx * cos_a - dy * sin_a
                    ly = dx * sin_a + dy * cos_a
                    for k in range(6):
                        theta = math.radians(30 + 60 * k)
                        proj = lx * math.cos(theta) + ly * math.sin(theta)
                        projs.append(proj)
            
            if not projs:
                return 0.0
            
            # Use log-sum-exp for stable soft-max
            m = max(projs)
            exp_sum = sum(math.exp(p - m) for p in projs)
            soft_max = m + math.log(exp_sum) / beta
            
            return soft_max * 2.0 / math.sqrt(3)
        
        def compute_gradient(inner_hexes, cx, cy, angle_deg, beta=50.0, eps=1e-5):
            """Compute numerical gradient of soft_max_outer_side."""
            f0 = soft_max_outer_side(inner_hexes, cx, cy, angle_deg, beta)
            
            dfdcx = (soft_max_outer_side(inner_hexes, cx + eps, cy, angle_deg, beta) -
                     soft_max_outer_side(inner_hexes, cx - eps, cy, angle_deg, beta)) / (2 * eps)
            
            dfdcy = (soft_max_outer_side(inner_hexes, cx, cy + eps, angle_deg, beta) -
                     soft_max_outer_side(inner_hexes, cx, cy - eps, angle_deg, beta)) / (2 * eps)
            
            dfdang = (soft_max_outer_side(inner_hexes, cx, cy, angle_deg + eps * 10, beta) -
                      soft_max_outer_side(inner_hexes, cx, cy, angle_deg - eps * 10, beta)) / (2 * eps * 10)
            
            return dfdcx, dfdcy, dfdang
        
        # Gradient descent with adaptive step size
        cur_cx, cur_cy, cur_ang = best_center[0], best_center[1], best_angle
        cur_side = soft_max_outer_side(inner_hexes, cur_cx, cur_cy, cur_ang, beta=50.0)
        
        # Use decreasing beta (increasing smoothness) for better gradient estimates
        for beta in [100.0, 50.0, 25.0, 10.0]:
            step = 0.1
            for _ in range(500):
                if time.time() > deadline - 10.0:
                    break
                
                gx, gy, ga = compute_gradient(inner_hexes, cur_cx, cur_cy, cur_ang, beta)
                grad_norm = math.sqrt(gx*gx + gy*gy + ga*ga)
                
                if grad_norm < 1e-8:
                    break
                
                # Normalize and apply step
                cur_cx -= step * gx / grad_norm * grad_norm
                cur_cy -= step * gy / grad_norm * grad_norm
                cur_ang -= step * ga / grad_norm * grad_norm
                
                new_side = soft_max_outer_side(inner_hexes, cur_cx, cur_cy, cur_ang, beta)
                
                if new_side < cur_side:
                    cur_side = new_side
                    if cur_side < best_side:
                        # Check actual (non-smooth) side length
                        actual_side = _min_outer_side(inner_hexes, cur_cx, cur_cy, cur_ang)
                        if actual_side < best_side:
                            best_side = actual_side
                            best_center = (cur_cx, cur_cy)
                            best_angle = cur_ang
                else:
                    # Backtrack
                    step *= 0.5
                    cur_cx += step * 2 * gx / grad_norm * grad_norm
                    cur_cy += step * 2 * gy / grad_norm * grad_norm
                    cur_ang += step * 2 * ga / grad_norm * grad_norm

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Final polish: high-precision optimization of outer hexagon parameters
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Use gradient-free local search with decreasing step sizes
        cur_cx, cur_cy = best_center
        cur_ang = best_angle
        cur_side = best_side
        
        for step_size in [0.05, 0.01, 0.002, 0.0005]:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try small perturbations
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        for da in [-step_size * 5, step_size * 5]:  # angle in degrees
                            ncx = cur_cx + dx
                            ncy = cur_cy + dy
                            nang = cur_ang + da
                            s = _min_outer_side(inner_hexes, ncx, ncy, nang)
                            if s < cur_side - 1e-12:
                                cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                                improved = True
                # Also try axis-aligned moves
                for dx, dy, da in [(-step_size, 0, 0), (step_size, 0, 0),
                                   (0, -step_size, 0), (0, step_size, 0),
                                   (0, 0, -step_size * 5), (0, 0, step_size * 5)]:
                    ncx = cur_cx + dx
                    ncy = cur_cy + dy
                    nang = cur_ang + da
                    s = _min_outer_side(inner_hexes, ncx, ncy, nang)
                    if s < cur_side - 1e-12:
                        cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                        improved = True
        
        if cur_side < best_side:
            best_side = cur_side
            best_center = (cur_cx, cur_cy)
            best_angle = cur_ang

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness - with proper axial spacing, this should always pass
    # The minimum distance between centers of adjacent hexagons is sqrt(3) ≈ 1.732
    # which is > 2*1.0 = 2.0? No, wait: for flat-top hexagons with side 1,
    # the distance between centers of adjacent hexagons should be sqrt(3) ≈ 1.732
    # But the hexagons have "radius" 1 (center to vertex), so they would overlap!
    
    # Actually, let me reconsider. For flat-top hexagons with side length 1:
    # - The distance from center to vertex is 1
    # - The distance from center to edge midpoint (apothem) is sqrt(3)/2 ≈ 0.866
    # - For two hexagons to not overlap, their centers must be far enough apart
    
    # In a hexagonal lattice with flat-top hexagons of side 1:
    # Adjacent hexagons share an edge. The distance between centers is 2 * apothem = sqrt(3) ≈ 1.732
    # But since the hexagons extend to radius 1 from center, and the shared edge is at apothem distance,
    # the hexagons actually share an edge and have disjoint interiors.
    
    # Let me verify: if two flat-top hexagons of side 1 are placed with centers distance sqrt(3) apart
    # horizontally, they share an edge. Their interiors are disjoint. Good.
    
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        # This shouldn't happen with correct axial mapping
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    # Verify all inner hexagons are inside outer with tight tolerance
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 14 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_angles = None  # per-hexagon rotation angles

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse grid to find promising configurations
    # Try multiple outer orientations per combo
    outer_angles_to_try = [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 10.0, -10.0, 20.0, -20.0]
    
    candidates = []  # list of (side, positions, center, angle)
    
    for combo in all_combos:
        if time.time() > deadline - 40.0:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        for oang in outer_angles_to_try:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
                best_inner_angles = [0.0] * 11
            candidates.append((side, full_xy, (cx_mean, cy_mean), oang))
    
    # Sort candidates and keep top few for fine-tuning
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:15]
    
    if best_positions is None and top_candidates:
        best_side, best_positions, best_center, best_angle = top_candidates[0]
        best_inner_angles = [0.0] * 11

    # Phase 2: Joint optimization of outer params AND inner rotations
    # Use soft-max surrogate for smoothness
    if best_positions is not None and time.time() < deadline:
        
        def soft_max_outer_side(positions, inner_angles, cx, cy, oangle_deg, beta=50.0):
            """Smooth approximation using log-sum-exp over all vertex projections."""
            a = -math.radians(oangle_deg)
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            
            all_verts = []
            for i, (px, py) in enumerate(positions):
                ang = inner_angles[i]
                for k in range(6):
                    theta = math.radians(ang + 60 * k)
                    vx = px + math.cos(theta)
                    vy = py + math.sin(theta)
                    all_verts.append((vx, vy))
            
            if not all_verts:
                return 0.0
            
            verts_arr = np.array(all_verts)
            dx = verts_arr[:, 0] - cx
            dy = verts_arr[:, 1] - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            projs = _OUTER_NORMALS_T @ np.column_stack([lx, ly])
            
            m = np.max(projs)
            exp_sum = np.sum(np.exp(projs - m))
            soft_max = m + math.log(max(exp_sum, 1e-300)) / beta
            return float(soft_max) * 2.0 / math.sqrt(3)
        
        def optimize_candidate(positions, inner_angles, cx, cy, oangle_deg):
            """Nelder-Mead style optimization over (cx, cy, oangle, inner_angles)."""
            if time.time() > deadline - 15.0:
                return None
            
            # Variables: [cx, cy, oangle, ang0, ang1, ..., ang10]
            x = np.array([cx, cy, oangle_deg] + list(inner_angles), dtype=np.float64)
            
            def objective(vars_):
                cx_v, cy_v, oang_v = vars_[0], vars_[1], vars_[2]
                in_angs = vars_[3:]
                return soft_max_outer_side(positions, in_angs, cx_v, cy_v, oang_v, beta=30.0)
            
            # Simple Nelder-Mead
            n = len(x)
            simplex = [x.copy()]
            for i in range(n):
                trial = x.copy()
                trial[i] += 0.05 if i < 3 else 1.0  # degrees for angles
                simplex.append(trial)
            
            fvals = [objective(s) for s in simplex]
            
            max_iter = 800
            for it in range(max_iter):
                if time.time() > deadline - 15.0:
                    break
                
                order = sorted(range(n + 1), key=lambda i: fvals[i])
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[-1] - fvals[0] < 1e-14:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = objective(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = objective(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = objective(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = objective(simplex[i])
            
            # Evaluate true (non-smooth) objective
            cx_v, cy_v, oang_v = simplex[0][0], simplex[0][1], simplex[0][2]
            in_angs = simplex[0][3:].tolist()
            inner_hexes = [(positions[i][0], positions[i][1], 1.0, in_angs[i]) for i in range(len(positions))]
            true_side = _min_outer_side(inner_hexes, cx_v, cy_v, oang_v)
            
            return (true_side, positions, (cx_v, cy_v), oang_v, in_angs)
        
        # Optimize top candidates
        for idx, (cand_side, cand_pos, cand_center, cand_angle) in enumerate(top_candidates):
            if time.time() > deadline - 20.0:
                break
            
            result = optimize_candidate(cand_pos, [0.0]*11, cand_center[0], cand_center[1], cand_angle)
            if result is not None:
                r_side, r_pos, r_center, r_angle, r_in_angs = result
                if r_side < best_side:
                    best_side = r_side
                    best_positions = r_pos
                    best_center = r_center
                    best_angle = r_angle
                    best_inner_angles = r_in_angs

    if best_positions is None:
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0
        best_inner_angles = [0.0] * 11

    # Phase 3: High-precision polish with exact objective
    if best_positions is not None and time.time() < deadline:
        if best_inner_angles is None:
            best_inner_angles = [0.0] * 11
        
        inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_inner_angles[i]) for i in range(11)]
        
        cur_cx, cur_cy = best_center
        cur_ang = best_angle
        cur_in_angs = list(best_inner_angles)
        cur_side = best_side
        
        # Coordinate descent with decreasing steps
        for step_size in [0.05, 0.02, 0.005, 0.001, 0.0002]:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                
                # Try perturbing outer center and angle
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        for da in [-step_size * 3, 0, step_size * 3]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            ncx = cur_cx + dx
                            ncy = cur_cy + dy
                            nang = cur_ang + da
                            ih = [(best_positions[i][0], best_positions[i][1], 1.0, cur_in_angs[i]) for i in range(11)]
                            s = _min_outer_side(ih, ncx, ncy, nang)
                            if s < cur_side - 1e-12:
                                cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                                improved = True
                
                # Try perturbing inner angles (affects shape orientation)
                for i in range(11):
                    for da in [-step_size * 5, step_size * 5]:
                        new_angs = list(cur_in_angs)
                        new_angs[i] += da
                        ih = [(best_positions[j][0], best_positions[j][1], 1.0, new_angs[j]) for j in range(11)]
                        s = _min_outer_side(ih, cur_cx, cur_cy, cur_ang)
                        if s < cur_side - 1e-12:
                            cur_in_angs = new_angs
                            cur_side = s
                            improved = True
        
        if cur_side < best_side:
            best_side = cur_side
            best_center = (cur_cx, cur_cy)
            best_angle = cur_ang
            best_inner_angles = cur_in_angs

    # Build final inner hexagons
    if best_inner_angles is None:
        best_inner_angles = [0.0] * 11
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_inner_angles[i]) for i in range(11)]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, best_angle)
                best_center = (cx_mean, cy_mean)
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
>>>>>>> REPLACE
