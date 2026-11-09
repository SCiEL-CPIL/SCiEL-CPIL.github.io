import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(points):
    """Normalize points: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return points / mean_dist

def _turn_signs(points):
    """Calculate the sign of the cross product for consecutive edges."""
    n = len(points)
    signs = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if cross > 1e-12:
            signs[i] = 1
        elif cross < -1e-12:
            signs[i] = -1
        else:
            signs[i] = 0
    return signs

def _is_valid(points):
    """Check if points form a strictly convex polygon in cyclic order."""
    if len(points) != 20:
        return False
    # Check for repeated points
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(points[i] - points[j]) < 1e-9:
                return False
    # Check convexity: all turns must have the same sign and be non-zero
    signs = _turn_signs(points)
    if np.any(signs == 0):
        return False
    if np.all(signs > 0) or np.all(signs < 0):
        return True
    return False

def _vertex_scores(points):
    """
    Compute the vertex score for each of the 20 vertices.
    Returns an array of 20 scores.
    """
    n = 20
    scores = np.zeros(n)
    # Precompute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        d = dists[i].copy()
        d[i] = np.inf
        sorted_d = np.sort(d)
        
        # Check: no distance < 0.05
        if sorted_d[0] < 0.05:
            scores[i] = 0.0
            continue
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = sorted_d[1:] - sorted_d[:-1]
        k = np.argmin(gaps)
        D = (sorted_d[k] + sorted_d[k + 1]) / 2.0
        
        # Check: D must be > 0.1
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = sorted_d[0:4]
        elif k == 17:
            d_sel = sorted_d[15:19]
        else:
            d_sel = sorted_d[k - 1:k + 3]
            
        # Calculate raw score
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _fourier_params_to_weights(fourier_params):
    """
    Convert 6 Fourier parameters to 20 positive angular weights.
    Params: [a1, b1, a2, b2, a3, b3]
    """
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a1, b1, a2, b2, a3, b3 = fourier_params
    
    deviations = np.zeros(n)
    for j in range(n):
        theta = 2 * np.pi * j / n
        deviations[j] = (a1 * np.cos(theta) + b1 * np.sin(theta) +
                         a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta) +
                         a3 * np.cos(3 * theta) + b3 * np.sin(3 * theta))
    
    angles = base_angles + deviations
    angles = np.sort(angles)
    
    # Compute weights (angular gaps)
    angles_full = np.append(angles, angles[0] + 2 * np.pi)
    weights = np.diff(angles_full)
    
    # Ensure positive and normalized
    weights = np.maximum(weights, 1e-8)
    weights = weights / np.sum(weights) * 2 * np.pi
    
    return weights

def _get_points_from_weights(weights):
    """Generate points on unit circle from angular weights."""
    angles = np.cumsum(weights)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def _get_points_from_fourier(fourier_params):
    """Generate normalized points from Fourier parameters."""
    w = _fourier_params_to_weights(fourier_params)
    pts = _get_points_from_weights(w)
    pts = _normalize(pts)
    if pts is None:
        return None
    return pts

def _soft_min(scores, beta):
    """
    Smooth minimum using log-sum-exp.
    Returns a value close to min(scores).
    """
    scores = np.asarray(scores, dtype=float)
    m = np.min(scores)
    if m < -1e10:
        return m
    # Use stable log-sum-exp
    exp_terms = np.exp(-beta * (scores - m))
    exp_terms = np.clip(exp_terms, 0, 1e10)
    return m - np.log(np.sum(exp_terms)) / beta

def _objective_fourier(params, beta=30.0):
    """
    Objective function for optimization.
    Returns negative of the soft-min of vertex scores.
    """
    pts = _get_points_from_fourier(params)
    if pts is None:
        return 1e6
    
    # Check validity
    if not _is_valid(pts):
        return 1e6
        
    scores = _vertex_scores(pts)
    if np.any(scores == 0):
        return 1e6 # If any vertex is invalid, score is 0, so objective is 0 (min score). We want to maximize min score, so minimize -min score.
    
    return -_soft_min(scores, beta)

def _objective_exact_fourier(params):
    """
    Exact objective function: negative of the minimum vertex score.
    """
    pts = _get_points_from_fourier(params)
    if pts is None:
        return 1e6
    
    if not _is_valid(pts):
        return 1e6
        
    scores = _vertex_scores(pts)
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Initial best: Regular polygon
    best_pts = _get_points_from_fourier(np.zeros(6))
    best_score = -_objective_exact_fourier(np.zeros(6))
    best_params = np.zeros(6)
    
    if best_pts is None:
        # Fallback, though regular polygon should be valid
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    # Phase 1: Differential Evolution for Global Search
    # Use a smooth objective (low beta) to help DE navigate
    if time.time() < deadline - 300:
        try:
            bounds = [(-0.5, 0.5) for _ in range(6)]
            # DE is slow, so limit iterations
            res_de = differential_evolution(
                _objective_fourier,
                bounds,
                args=(5.0,), # Low beta for smoothness
                maxiter=50,
                popsize=15,
                tol=1e-6,
                update_interval=1,
                seed=42,
                polish=False # We will polish manually
            )
            s = -_objective_exact_fourier(res_de.x)
            if s > best_score:
                best_score = s
                best_params = res_de.x.copy()
                best_pts = _get_points_from_fourier(res_de.x)
        except Exception:
            pass

    # Phase 2: Multi-start Local Refinement with Annealing
    # Generate starting points
    rng = np.random.default_rng(42)
    starts = [best_params.copy()]
    
    # Random starts
    for i in range(20):
        if time.time() > deadline - 150:
            break
        x_rand = rng.uniform(-0.3, 0.3, size=6)
        starts.append(x_rand)
        
    # Perturbed best
    for i in range(10):
        if time.time() > deadline - 150:
            break
        x_pert = best_params + rng.normal(0, 0.05, size=6)
        x_pert = np.clip(x_pert, -0.5, 0.5)
        starts.append(x_pert)
        
    # Beta progression: 10 -> 20 -> 50 -> 100
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 50:
            break
            
        for x_init in starts:
            if time.time() > deadline - 50:
                break
            try:
                res_lm = minimize(
                    _objective_fourier,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5) for _ in range(6)],
                    args=(beta,),
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                s = -_objective_exact_fourier(res_lm.x)
                if s > best_score:
                    best_score = s
                    best_params = res_lm.x.copy()
                    best_pts = _get_points_from_fourier(res_lm.x)
                    # Add improved solution to starts for next beta
                    starts.append(res_lm.x.copy())
            except Exception:
                pass

    # Phase 3: High-Precision Polish on Exact Objective
    # Use Nelder-Mead which is derivative-free and handles non-smoothness better
    if time.time() < deadline - 10:
        # Start from best found
        try:
            res_nm = minimize(
                _objective_exact_fourier,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -_objective_exact_fourier(res_nm.x)
            if s > best_score:
                best_score = s
                best_params = res_nm.x.copy()
                best_pts = _get_points_from_fourier(res_nm.x)
        except Exception:
            pass
        
        # Try a few more restarts around the best
        for i in range(5):
            if time.time() > deadline - 5:
                break
            x_pert = best_params + rng.normal(0, 0.01, size=6)
            x_pert = np.clip(x_pert, -0.5, 0.5)
            try:
                res_nm = minimize(
                    _objective_exact_fourier,
                    x_pert,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = -_objective_exact_fourier(res_nm.x)
                if s > best_score:
                    best_score = s
                    best_params = res_nm.x.copy()
                    best_pts = _get_points_from_fourier(res_nm.x)
            except Exception:
                pass

    # Final validation and return
    if best_pts is None or not _is_valid(best_pts):
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts
