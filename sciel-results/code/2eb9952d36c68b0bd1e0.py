import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        dists[i] = np.inf
        dists = np.sort(dists)
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _normalize(polygon):
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
    if mean_dist <= 0.01:
        return None
    return polygon_norm / mean_dist

def _is_strictly_convex(polygon):
    v_curr = polygon
    v_next = np.roll(polygon, -1, axis=0)
    v_next2 = np.roll(polygon, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return np.all(cross > 1e-8)

def _params_to_polygon(params):
    """
    Convert edge lengths and exterior angles to polygon vertices.
    params: [e0, e1, ..., e19, a0, a1, ..., a18] (39 params)
    The 19th exterior angle is determined by closure: a19 = 2*pi - sum(a0..a18)
    All exterior angles must be positive for convexity.
    """
    n = 20
    edge_lengths = params[:n]
    ext_angles = params[n:n+19]  # 19 angles
    # 20th angle from closure
    a19 = 2 * np.pi - np.sum(ext_angles)
    all_angles = np.concatenate([ext_angles, [a19]])
    
    # If any angle is non-positive, polygon is not convex
    if np.any(all_angles <= 0):
        return None
    
    # Build polygon by walking along edges with turning angles
    vertices = np.zeros((n, 2))
    vertices[0] = [0.0, 0.0]
    current_angle = 0.0
    
    for i in range(n):
        if i > 0:
            # Turn by exterior angle at vertex i-1
            current_angle += all_angles[i-1]
        # Edge i goes from vertex i to vertex i+1
        dx = edge_lengths[i] * np.cos(current_angle)
        dy = edge_lengths[i] * np.sin(current_angle)
        if i < n - 1:
            vertices[i+1] = vertices[i] + np.array([dx, dy])
    
    # Check closure
    closure_error = np.linalg.norm(vertices[0] - (vertices[-1] + np.array([
        edge_lengths[-1] * np.cos(current_angle),
        edge_lengths[-1] * np.sin(current_angle)
    ])))
    
    if closure_error > 0.5:  # Too much closure error
        return None
    
    # Close the polygon
    vertices[0] = np.mean(vertices, axis=0)  # Center it
    return vertices

def _params_to_polygon_v2(params):
    """
    Alternative: use 20 angles (one per vertex) and 20 edge lengths.
    The sum of exterior angles must be 2*pi for closure.
    params: [e0..e19, a0..a19] where sum(a) = 2*pi
    We optimize 39 free params and set a19 = 2*pi - sum(a0..a18)
    """
    n = 20
    edge_lengths = params[:n]
    ext_angles_19 = params[n:n+19]
    a19 = 2 * np.pi - np.sum(ext_angles_19)
    
    if np.any(ext_angles_19 <= 0) or a19 <= 0:
        return None
    
    all_angles = np.concatenate([ext_angles_19, [a19]])
    
    # Walk around the polygon
    vertices = np.zeros((n, 2))
    vertices[0] = [0.0, 0.0]
    heading = 0.0
    
    for i in range(n):
        if i == 0:
            heading = 0.0
        else:
            heading += all_angles[i-1]
        
        if i < n - 1:
            vertices[i+1] = vertices[i] + edge_lengths[i] * np.array([np.cos(heading), np.sin(heading)])
        else:
            # Last edge should close to vertex 0
            pass
    
    # The last vertex position after walking all edges
    last_pos = vertices[n-1] + edge_lengths[n-1] * np.array([np.cos(heading), np.sin(heading)])
    closure_err = np.linalg.norm(last_pos - vertices[0])
    
    if closure_err > 1.0:
        return None
    
    # For valid closure, the polygon is closed. Center it.
    centroid = np.mean(vertices, axis=0)
    return vertices - centroid

def _objective(params):
    polygon = _params_to_polygon_v2(params)
    if polygon is None:
        return 1000.0
    
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    # Check convexity
    if not _is_strictly_convex(polygon_norm):
        # Penalty for non-convexity
        v_curr = polygon_norm
        v_next = np.roll(polygon_norm, -1, axis=0)
        v_next2 = np.roll(polygon_norm, -2, axis=0)
        e1 = v_next - v_curr
        e2 = v_next2 - v_next
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        neg_cross = np.minimum(cross, 0)
        penalty = np.sum(neg_cross**2)
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 - 15.0  # Safety margin
    
    n = 20
    # Regular polygon as fallback
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # For regular polygon: all edge lengths = 2*sin(pi/20), all exterior angles = pi/10
    edge_len_reg = 2 * np.sin(np.pi / 20)
    ext_angle_reg = np.pi / 10
    
    # Initial params: 20 edge lengths + 19 exterior angles
    x0 = np.concatenate([np.full(n, edge_len_reg), np.full(19, ext_angle_reg)])
    
    best_polygon = regular.copy()
    best_score = -1e9
    best_params = x0.copy()
    
    # Bounds: edge lengths in [0.1, 3.0], exterior angles in [0.01, 0.5]
    # (exterior angle max ~0.5 rad ~ 28.6 degrees, which is reasonable for convex 20-gon)
    bounds = [(0.1, 3.0)] * n + [(0.01, 0.5)] * 19
    
    # Phase 1: Differential Evolution for global search
    try:
        if time.time() < deadline - 120:
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=300,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < 100:  # Valid result
                best_params = de_result.x
                pn = _params_to_polygon_v2(best_params)
                if pn is not None:
                    pn_norm = _normalize(pn)
                    if pn_norm is not None and _is_strictly_convex(pn_norm):
                        s = np.min(compute_scores(pn_norm))
                        if s > best_score:
                            best_score = s
                            best_polygon = pn_norm
    except Exception:
        pass
    
    # Phase 2: Local polish with Nelder-Mead
    try:
        if time.time() < deadline - 30:
            if best_score > 0:
                x0_polish = best_params
            else:
                x0_polish = x0
            
            polish_result = minimize(
                _objective, x0_polish,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if polish_result.fun < 100:
                pn = _params_to_polygon_v2(polish_result.x)
                if pn is not None:
                    pn_norm = _normalize(pn)
                    if pn_norm is not None and _is_strictly_convex(pn_norm):
                        s = np.min(compute_scores(pn_norm))
                        if s > best_score:
                            best_score = s
                            best_polygon = pn_norm
                            best_params = polish_result.x
    except Exception:
        pass
    
    # Phase 3: Additional local refinement if time allows
    try:
        if time.time() < deadline - 10 and best_score > 0.5:
            final_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if final_result.fun < 100:
                pn = _params_to_polygon_v2(final_result.x)
                if pn is not None:
                    pn_norm = _normalize(pn)
                    if pn_norm is not None and _is_strictly_convex(pn_norm):
                        s = np.min(compute_scores(pn_norm))
                        if s > best_score:
                            best_score = s
                            best_polygon = pn_norm
    except Exception:
        pass
    
    # Return the best valid polygon
    if best_score > 0:
        pn = _normalize(best_polygon)
        if pn is not None and _is_strictly_convex(pn):
            return pn
    
    # Fallback to regular polygon
    return regular
