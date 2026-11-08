"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_verts_fast(cx, cy, side, angle_deg):
    """Fast hexagon vertices using precomputed cos/sin."""
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

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

def _compute_all_penalties(params, outer_side, outer_center, outer_angle):
    """Compute all inner hex vertices and return containment + disjointness penalties."""
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
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
    
    return cont_pen + disj_pen, inner_verts_list

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
    pen, _ = _compute_all_penalties(params, outer_side, outer_center, outer_angle)
    return pen

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
    
    # Generate candidate initial configurations
    configs = []
    
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
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
    
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Quick L-BFGS-B search with multiple outer angles and sides
    outer_angles = [0.0, 30.0]
    outer_sides = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]
    
    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 30.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 25.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                pen = _total_penalty(init_params, outer_side, outer_center, outer_angle)
                
                if pen < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue
                
                if time.time() > deadline - 25.0:
                    continue
                
                def objective(p):
                    return _total_penalty(p, outer_side, outer_center, outer_angle)
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'maxfun': 2500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Phase 2: Differential Evolution for global exploration
    # Use a larger container to allow DE to find good configurations
    de_outer_side = 4.5
    de_outer_angle = 0.0
    de_outer_center = (0.0, 0.0)
    
    # Bounds: positions in [-3, 3], angles in [0, 60]
    bounds = []
    for _ in range(N):
        bounds.extend([(-3.0, 3.0), (-3.0, 3.0), (0.0, 60.0)])
    
    def de_objective(params):
        return _total_penalty(params, de_outer_side, de_outer_center, de_outer_angle)
    
    if time.time() < deadline - 40.0:
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            de_params = de_result.x
            de_val = de_result.fun
            
            if de_val < 1e-6:
                # DE found a valid configuration in the 4.5-side container
                # Now try to shrink the container
                current_params = de_params.copy()
                current_side = de_outer_side
                current_angle = de_outer_angle
                
                # Binary search for minimum side length
                low = 3.0
                high = current_side
                
                for _ in range(8):
                    if time.time() > deadline - 15.0:
                        break
                    mid = (low + high) / 2.0
                    if mid <= low + 1e-6:
                        break
                    
                    def shrink_obj(p, s=mid):
                        return _total_penalty(p, s, de_outer_center, de_outer_angle)
                    
                    try:
                        shrink_result = minimize(
                            shrink_obj,
                            current_params,
                            method='L-BFGS-B',
                            options={'maxiter': 50, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if shrink_result.fun < 1e-6:
                            high = mid
                            current_params = shrink_result.x
                            current_side = mid
                        else:
                            low = mid
                    except Exception:
                        low = mid
                
                if current_side < best_score:
                    best_score = current_side
                    best_result = (current_params, de_outer_center, current_side, current_angle)
                    
        except Exception:
            pass
    
    # Phase 3: Final refinement — binary search for minimal valid side length
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Binary search for the smallest side length that still admits a valid packing
        low = max(2.8, outer_side - 0.5)
        high = outer_side
        
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            mid = (low + high) / 2.0
            if high - low < 1e-5:
                break
            
            try:
                result = minimize(
                    lambda p, s=mid: _total_penalty(p, s, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-8:
                    high = mid
                    opt_params = result.x
                    outer_side = mid
                else:
                    low = mid
            except Exception:
                low = mid
        
        # Final high-precision polish at the found side length
        if time.time() < deadline - 3.0:
            try:
                result = minimize(
                    lambda p: _total_penalty(p, outer_side, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < 1e-8:
                    opt_params = result.x
            except Exception:
                pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
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
