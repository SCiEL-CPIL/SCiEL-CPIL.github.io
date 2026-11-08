"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, verts):
    """Check if point is inside polygon using winding number / cross products."""
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -1e-10:
            return False
    return True

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _check_valid(params, outer_side, outer_center, outer_angle, tol=1e-6):
    """Check if configuration is valid (all inner hexagons inside outer, no overlaps)."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check containment
    for iv in inner_verts_list:
        for pt in iv:
            if not _point_in_polygon(pt, outer_verts):
                return False
    
    # Check disjointness
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > tol:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]
    
    def build_compact_config():
        """Compact 2x6 arrangement with slight rotation."""
        positions = []
        for r in range(2):
            for c in range(6):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        arr = np.array(positions)
        centroid = arr.mean(axis=0)
        return (arr - centroid).tolist()
    
    # Generate candidate initial configurations
    configs = []
    
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    compact = build_compact_config()
    configs.append(compact)
    
    best_score = float('inf')
    best_result = None
    
    def try_local_optimize(init_params, outer_side, outer_center, outer_angle, max_iter=80, maxfun=2000):
        """Try local optimization and return (success, params, penalty)."""
        def objective(p):
            return _total_penalty(p, outer_side, outer_center, outer_angle)
        
        try:
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': maxfun, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            return True, result.x, result.fun
        except Exception:
            return False, None, float('inf')
    
    def try_de_global(outer_side, outer_center, outer_angle, maxiter=30):
        """Try differential evolution for global search."""
        if time.time() > deadline - 20.0:
            return None
        
        # Bounds: x, y in [-5, 5], angle in [0, 180]
        bounds = [(-5, 5), (-5, 5), (0, 180)] * N
        
        def de_objective(p):
            return _total_penalty(p, outer_side, outer_center, outer_angle)
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=maxiter,
                popsize=10,
                tol=1e-8,
                seed=42,
                polish=True,
                options={'maxiter': maxiter}
            )
            if result.success and result.fun < 1e-6:
                return result.x
            elif result.fun < 0.01:
                return result.x
        except Exception:
            pass
        return None
    
    # Phase 1: Try local optimization from multiple starts with decreasing outer sides
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 30.0:
            break
        
        outer_center = (0.0, 0.0)
        
        # Try a range of outer sides from large to small
        outer_sides = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2]
        
        for outer_side in outer_sides:
            if time.time() > deadline - 25.0:
                break
            
            found_valid = False
            for config in configs:
                if time.time() > deadline - 20.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                # Quick check
                pen = _total_penalty(init_params, outer_side, outer_center, outer_angle)
                
                if pen < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    found_valid = True
                    continue
                
                # Local optimization
                success, opt_params, opt_val = try_local_optimize(
                    init_params, outer_side, outer_center, outer_angle
                )
                
                if success and opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
                    found_valid = True
                    continue
                
                # If close, try with slightly smaller outer side
                if success and opt_val < 0.05 and outer_side > 3.2:
                    smaller = outer_side - 0.1
                    if time.time() > deadline - 15.0:
                        continue
                    success2, opt_params2, opt_val2 = try_local_optimize(
                        opt_params, smaller, outer_center, outer_angle, max_iter=60, maxfun=1500
                    )
                    if success2 and opt_val2 < 1e-6:
                        if smaller < best_score:
                            best_score = smaller
                            best_result = (opt_params2, outer_center, smaller, outer_angle)
                        found_valid = True
    
    # Phase 2: If we have a result, try to improve with DE or finer search
    if best_result is not None and time.time() < deadline - 15.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Try binary search for smaller outer side
        low = outer_side - 0.3
        high = outer_side
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            mid = (low + high) / 2
            if mid < 3.0:
                break
            
            success, opt_params_new, opt_val = try_local_optimize(
                opt_params, mid, outer_center, outer_angle, max_iter=60, maxfun=1500
            )
            
            if success and opt_val < 1e-6:
                high = mid
                opt_params = opt_params_new
                outer_side = mid
                if outer_side < best_score:
                    best_score = outer_side
                    best_result = (opt_params, outer_center, outer_side, outer_angle)
            else:
                low = mid
        
        # Try DE if time permits
        if time.time() < deadline - 25.0 and outer_side > 3.1:
            de_params = try_de_global(outer_side - 0.2, outer_center, outer_angle, maxiter=20)
            if de_params is not None:
                # Verify and try to refine
                success, opt_params_de, opt_val_de = try_local_optimize(
                    de_params, outer_side - 0.2, outer_center, outer_angle, max_iter=40, maxfun=1000
                )
                if success and opt_val_de < 1e-6:
                    new_side = outer_side - 0.2
                    if new_side < best_score:
                        best_score = new_side
                        best_result = (opt_params_de, outer_center, new_side, outer_angle)
    
    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = sum(_containment_penalty(iv, outer_verts) for iv in inner_verts_list)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
