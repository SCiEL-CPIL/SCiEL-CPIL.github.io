import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)

def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        # Check for degeneracy
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_objective_structured(params, template_id, beta=50.0):
    """
    Objective function for structured templates.
    Returns negative soft-min of triangle area ratios.
    """
    # Build points from params
    if template_id == 0: # 13-0
        r = params[0]
        a0 = params[1]
        angles = np.linspace(a0, 2 * np.pi + a0, N, endpoint=False)
        points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    elif template_id == 1: # 7-6
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 7, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 6, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 2: # 6-7
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 6, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 7, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 3: # 5-4-4
        r_o, r_m, r_i, a_o, a_m, a_i = params
        o_angles = np.linspace(a_o, 2 * np.pi + a_o, 5, endpoint=False)
        m_angles = np.linspace(a_m, 2 * np.pi + a_m, 4, endpoint=False)
        i_angles = np.linspace(a_i, 2 * np.pi + a_i, 4, endpoint=False)
        o_pts = np.column_stack([r_o * np.cos(o_angles), r_o * np.sin(o_angles)])
        m_pts = np.column_stack([r_m * np.cos(m_angles), r_m * np.sin(m_angles)])
        i_pts = np.column_stack([r_i * np.cos(i_angles), r_i * np.sin(i_angles)])
        points = np.vstack([o_pts, m_pts, i_pts])
    elif template_id == 4: # 8-5
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 8, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 5, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 5: # 9-4
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 9, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 4, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 6: # 10-3
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 10, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 3, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 7: # 11-2
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 11, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 2, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    elif template_id == 8: # 12-1
        r_out, r_in, a_out, a_in = params
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 12, endpoint=False)
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 1, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        points = np.vstack([outer_pts, inner_pts])
    else:
        return 1e10

    if not np.all(np.isfinite(points)):
        return 1e10

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    # Soft min
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    
    return -soft_min_val

def _negative_score_full(x_flat):
    """Objective for full 26D optimization."""
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Define template initial parameters
    # We will try multiple random starts for each template
    
    # Template definitions: (id, n_params, init_params_list)
    # We use a fixed outer radius of 1.0 to remove scale invariance, 
    # but we allow the optimizer to adjust it slightly if needed? 
    # Actually, scale invariance means we can fix one radius. 
    # Let's fix the largest radius to 1.0. 
    # In the builders above, I didn't fix it. 
    # To be safe, I will add bounds and let the optimizer find the scale, 
    # but I'll initialize with reasonable values.
    
    templates = [
        (0, 2), # 13-0: [r, offset]
        (1, 4), # 7-6: [r_out, r_in, a_out, a_in]
        (2, 4), # 6-7
        (3, 6), # 5-4-4
        (4, 4), # 8-5
        (5, 4), # 9-4
        (6, 4), # 10-3
        (7, 4), # 11-2
        (8, 4), # 12-1
    ]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Structured Optimization
    for tid, n_params in templates:
        if time.time() > deadline - 10:
            break
            
        # Try 5 random restarts for this template
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize parameters
            if tid == 0:
                # Regular 13-gon
                init_params = np.array([1.0, 0.0])
            elif tid == 1: # 7-6
                init_params = np.array([1.0, 0.5, 0.0, np.pi/6 + rng.uniform(-0.1, 0.1)])
            elif tid == 2: # 6-7
                init_params = np.array([1.0, 0.5, 0.0, np.pi/7 + rng.uniform(-0.1, 0.1)])
            elif tid == 3: # 5-4-4
                init_params = np.array([1.0, 0.6, 0.3, 0.0, np.pi/4, np.pi/4 + rng.uniform(-0.1, 0.1)])
            elif tid == 4: # 8-5
                init_params = np.array([1.0, 0.5, 0.0, np.pi/5 + rng.uniform(-0.1, 0.1)])
            elif tid == 5: # 9-4
                init_params = np.array([1.0, 0.5, 0.0, np.pi/4 + rng.uniform(-0.1, 0.1)])
            elif tid == 6: # 10-3
                init_params = np.array([1.0, 0.5, 0.0, np.pi/3 + rng.uniform(-0.1, 0.1)])
            elif tid == 7: # 11-2
                init_params = np.array([1.0, 0.5, 0.0, 0.0 + rng.uniform(-0.1, 0.1)])
            elif tid == 8: # 12-1
                init_params = np.array([1.0, 0.0, 0.0, 0.0 + rng.uniform(-0.1, 0.1)])
            else:
                init_params = np.random.uniform(0.1, 1.0, n_params)
            
            # Add some randomness to the init to explore
            if restart > 0:
                noise = rng.uniform(-0.1, 0.1, n_params)
                # Keep radii positive
                init_params = init_params + noise
                init_params[0] = np.abs(init_params[0]) + 0.1
                if n_params > 1:
                    init_params[1] = np.abs(init_params[1]) + 0.1
                if n_params > 2:
                    # Angles can be anything
                    pass
                
            # Optimize
            try:
                result = minimize(
                    _soft_min_objective_structured,
                    init_params,
                    args=(tid,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                # Reconstruct points from optimized params
                opt_params = result.x
                # We need to rebuild the points using the same logic as in the objective
                # Let's create a helper to build points from params and tid
                if tid == 0:
                    r, a0 = opt_params
                    angles = np.linspace(a0, 2 * np.pi + a0, N, endpoint=False)
                    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                elif tid in [1, 2, 4, 5, 6, 7, 8]:
                    r_out, r_in, a_out, a_in = opt_params
                    if tid == 1: n_out, n_in = 7, 6
                    elif tid == 2: n_out, n_in = 6, 7
                    elif tid == 4: n_out, n_in = 8, 5
                    elif tid == 5: n_out, n_in = 9, 4
                    elif tid == 6: n_out, n_in = 10, 3
                    elif tid == 7: n_out, n_in = 11, 2
                    elif tid == 8: n_out, n_in = 12, 1
                    
                    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, n_out, endpoint=False)
                    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, n_in, endpoint=False)
                    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
                    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
                    pts = np.vstack([outer_pts, inner_pts])
                elif tid == 3:
                    r_o, r_m, r_i, a_o, a_m, a_i = opt_params
                    o_angles = np.linspace(a_o, 2 * np.pi + a_o, 5, endpoint=False)
                    m_angles = np.linspace(a_m, 2 * np.pi + a_m, 4, endpoint=False)
                    i_angles = np.linspace(a_i, 2 * np.pi + a_i, 4, endpoint=False)
                    o_pts = np.column_stack([r_o * np.cos(o_angles), r_o * np.sin(o_angles)])
                    m_pts = np.column_stack([r_m * np.cos(m_angles), r_m * np.sin(m_angles)])
                    i_pts = np.column_stack([r_i * np.cos(i_angles), r_i * np.sin(i_angles)])
                    pts = np.vstack([o_pts, m_pts, i_pts])
                else:
                    continue
                    
                _update_best(pts)
                
            except Exception:
                pass

    # Phase 2: Full Space Polish
    # If we found a good candidate, polish it in the full 26D space
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _negative_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            _update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
