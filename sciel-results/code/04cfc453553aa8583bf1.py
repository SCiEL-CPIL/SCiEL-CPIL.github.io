import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_scores(polygon):
    """Compute the vertex score for each vertex of the polygon."""
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

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts / mean_dist

def _is_valid(pts):
    n = len(pts)
    if n != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # Check convexity
    v_curr = pts
    v_next = np.roll(pts, -1, axis=0)
    v_next2 = np.roll(pts, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(cross <= 1e-6):
        return False
    return True

def _min_vertex_score(pts):
    if pts is None:
        return 0.0
    if not _is_valid(pts):
        return 0.0
    scores = compute_scores(pts)
    return np.min(scores)

def _fourier_radial_to_polygon(coeffs, n_harmonics=4, n_samples=20):
    """
    Convert Fourier coefficients of radial function to polygon vertices.
    r(theta) = 1 + sum_{k=1}^{n_h} (a_k cos(k*theta) + b_k sin(k*theta))
    """
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    r = np.ones(n_samples)
    for k in range(1, n_harmonics + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1) + 1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure radius is positive
    r = np.maximum(r, 0.3)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _objective_radial(params, n_harmonics=4):
    """Objective for radial function optimization."""
    n_samples = 20
    polygon = _fourier_radial_to_polygon(params, n_harmonics, n_samples)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    # Check convexity penalty
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def _soft_min_objective(params, temperature=0.05, n_harmonics=4):
    """Smooth surrogate using log-sum-exp for the min score."""
    n_samples = 20
    polygon = _fourier_radial_to_polygon(params, n_harmonics, n_samples)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    scores_clipped = np.maximum(scores, 1e-10)
    neg_scores = -scores_clipped
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min_val = (1.0 / temperature) * log_sum
    
    return soft_min_val

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n_harmonics = 4
    n_params = 2 * n_harmonics
    
    # Initial best: regular polygon
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    r_init = np.ones(20)
    x_init = r_init * np.cos(angles)
    y_init = r_init * np.sin(angles)
    poly_init = np.column_stack((x_init, y_init))
    poly_init_norm = _normalize(poly_init)
    best_score = _min_vertex_score(poly_init_norm)
    best_poly = poly_init_norm.copy()
    
    if best_score == 0.0:
        # Fallback if regular polygon is invalid for some reason
        best_score = 0.0
        best_poly = poly_init_norm
    
    # Differential Evolution
    bounds = [(-0.2, 0.2)] * n_params
    
    def de_objective(params):
        if time.time() > deadline - 10:
            return 1000.0
        return _objective_radial(params, n_harmonics)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            update_interval=1
        )
        
        if result.success or result.fun < -best_score:
            best_params = result.x
            poly_candidate = _fourier_radial_to_polygon(best_params, n_harmonics, 20)
            poly_candidate_norm = _normalize(poly_candidate)
            cand_score = _min_vertex_score(poly_candidate_norm)
            if cand_score > best_score:
                best_score = cand_score
                best_poly = poly_candidate_norm
    except Exception as e:
        pass
    
    # Nelder-Mead Polish
    if time.time() < deadline - 30:
        # Initialize Nelder-Mead from best DE result or zero
        if 'best_params' in locals():
            nm_init = best_params
        else:
            nm_init = np.zeros(n_params)
            
        try:
            result_nm = minimize(
                _soft_min_objective,
                nm_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            
            if result_nm.success or result_nm.fun < -best_score * 10: # rough check
                best_params_nm = result_nm.x
                poly_candidate = _fourier_radial_to_polygon(best_params_nm, n_harmonics, 20)
                poly_candidate_norm = _normalize(poly_candidate)
                cand_score = _min_vertex_score(poly_candidate_norm)
                if cand_score > best_score:
                    best_score = cand_score
                    best_poly = poly_candidate_norm
        except Exception as e:
            pass
    
    # Final high-precision polish if time allows
    if time.time() < deadline - 10 and best_score > 0:
        # Try a few random restarts around the best point
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            noise = np.random.normal(0, 0.01, n_params)
            perturbed_params = best_params + noise if 'best_params' in locals() else noise
            poly_candidate = _fourier_radial_to_polygon(perturbed_params, n_harmonics, 20)
            poly_candidate_norm = _normalize(poly_candidate)
            cand_score = _min_vertex_score(poly_candidate_norm)
            if cand_score > best_score:
                best_score = cand_score
                best_poly = poly_candidate_norm
                best_params = perturbed_params

    return best_poly
