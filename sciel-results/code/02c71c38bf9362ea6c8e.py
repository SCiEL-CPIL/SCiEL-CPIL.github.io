import numpy as np
import time
from scipy.optimize import minimize

def _compute_all_scores(polygon):
    """Compute per-vertex scores for all vertices of polygon."""
    n = len(polygon)
    scores = np.zeros(n)
    diff = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        d = np.sort(dists[i])
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _polygon_min_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    return float(np.min(_compute_all_scores(polygon)))

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    dists = np.sqrt(np.sum(poly**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return poly
    return poly / mean_dist

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex with correct total turning."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    cross_prods = []
    for i in range(n):
        v1 = polygon[(i+1) % n] - polygon[i]
        v2 = polygon[(i+2) % n] - polygon[(i+1) % n]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    edge_angles = []
    for i in range(n):
        v = polygon[(i+1) % n] - polygon[i]
        edge_angles.append(np.arctan2(v[1], v[0]))
    turns = []
    for i in range(n):
        a1 = edge_angles[i]
        a2 = edge_angles[(i+1) % n]
        turn = a2 - a1
        while turn > np.pi:
            turn -= 2*np.pi
        while turn <= -np.pi:
            turn += 2*np.pi
        turns.append(turn)
    total_turn = sum(turns)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return False
    return True

def _build_polygon_from_radii(radii, angles):
    """Build polygon from radii and fixed angles."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _build_polygon_fourier(params, n=20):
    """Build polygon from radial Fourier modulation parameters."""
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = np.full(n, params[0])
    idx = 1
    for m in range(1, 6):
        a_m = params[idx]
        b_m = params[idx+1]
        r += a_m * np.cos(m * theta) + b_m * np.sin(m * theta)
        idx += 2
    r = np.maximum(r, 0.05)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def _softmin(scores, beta=50.0):
    """Soft minimum via log-sum-exp."""
    return -np.log(np.sum(np.exp(-beta * scores)) / len(scores)) / beta

def _softmin_objective(polygon, beta=50.0):
    """Negative soft-min vertex score (smooth surrogate)."""
    scores = _compute_all_scores(polygon)
    if np.any(scores < 0):
        return 10.0
    return -_softmin(scores, beta)

def _hard_objective(polygon):
    """Negative hard min vertex score."""
    return -_polygon_min_score(polygon)

def _evaluate_params(params, angles, n=20):
    """Build polygon from radii params, normalize, check convexity, return negative min score."""
    polygon = _build_polygon_from_radii(params, angles)
    polygon = _normalize(polygon)
    if not _is_strictly_convex(polygon):
        return 10.0
    return -_polygon_min_score(polygon)

def _evaluate_params_soft(params, angles, n=20, beta=50.0):
    """Soft-min version for gradient-based methods."""
    polygon = _build_polygon_from_radii(params, angles)
    polygon = _normalize(polygon)
    if not _is_strictly_convex(polygon):
        return 10.0
    scores = _compute_all_scores(polygon)
    if np.any(scores < 0):
        return 10.0
    return -_softmin(scores, beta)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum per-vertex equidistance score.
    Uses Fourier parameterization, L-BFGS-B with soft-min surrogate, and Nelder-Mead refinement.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    n = 20
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_polygon = _normalize(best_polygon)
    best_score = _polygon_min_score(best_polygon)
    
    np.random.seed(42)
    
    def time_left():
        return deadline - time.time()
    
    # Strategy 1: L-BFGS-B with soft-min surrogate in Fourier space
    n_params = 1 + 2*5
    base_params = np.zeros(n_params)
    base_params[0] = 1.0
    
    def _fourier_soft_objective(params, beta=30.0):
        polygon = _build_polygon_fourier(params, n)
        polygon = _normalize(polygon)
        if not _is_strictly_convex(polygon):
            return 10.0
        scores = _compute_all_scores(polygon)
        if np.any(scores < 0):
            return 10.0
        return -_softmin(scores, beta)
    
    def _fourier_hard_objective(params):
        polygon = _build_polygon_fourier(params, n)
        polygon = _normalize(polygon)
        if not _is_strictly_convex(polygon):
            return 10.0
        return -_polygon_min_score(polygon)
    
    # Phase 1a: L-BFGS-B with moderate soft-min
    try:
        if time_left() > 150:
            result = minimize(
                _fourier_soft_objective, base_params.copy(), method='L-BFGS-B',
                args=(20.0,), options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if time_left() > 100:
                # Refine with tighter soft-min
                result2 = minimize(
                    _fourier_soft_objective, result.x.copy(), method='L-BFGS-B',
                    args=(80.0,), options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                poly = _normalize(_build_polygon_fourier(result2.x, n))
                if _is_strictly_convex(poly):
                    s = _polygon_min_score(poly)
                    if s > best_score:
                        best_score = s
                        best_polygon = poly
    except Exception:
        pass
    
    # Phase 1b: L-BFGS-B from random perturbations
    for restart in range(8):
        if time_left() < 120:
            break
        params = base_params.copy()
        scale = 0.12 / (1 + restart * 0.2)
        for i in range(1, n_params):
            params[i] += np.random.uniform(-scale, scale)
        
        try:
            result = minimize(
                _fourier_soft_objective, params.copy(), method='L-BFGS-B',
                args=(25.0,), options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if time_left() < 80:
                break
            # Tighten
            result2 = minimize(
                _fourier_soft_objective, result.x.copy(), method='L-BFGS-B',
                args=(100.0,), options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            poly = _normalize(_build_polygon_fourier(result2.x, n))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead on hard objective (from best Fourier result)
    if time_left() > 100:
        # Get current best radii
        bp_radii = np.sqrt(np.sum(best_polygon**2, axis=1))
        
        try:
            result = minimize(
                _evaluate_params, bp_radii.copy(), method='Nelder-Mead',
                args=(angles, n), options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            poly = _normalize(_build_polygon_from_radii(result.x, angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass
    
    # Strategy 3: Multi-restart Nelder-Mead on radii
    for restart in range(10):
        if time_left() < 80:
            break
        bp = best_polygon
        current_radii = np.sqrt(np.sum(bp**2, axis=1)).copy()
        # Perturb
        perturb_scale = 0.05 / (1 + restart * 0.3)
        trial_radii = current_radii + np.random.uniform(-perturb_scale, perturb_scale, n)
        trial_radii = np.maximum(trial_radii, 0.1)
        
        try:
            result = minimize(
                _evaluate_params, trial_radii.copy(), method='Nelder-Mead',
                args=(angles, n), options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            poly = _normalize(_build_polygon_from_radii(result.x, angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass
    
    # Strategy 4: L-BFGS-B on radii with soft-min (fine-tune)
    if time_left() > 60:
        bp_radii = np.sqrt(np.sum(best_polygon**2, axis=1)).copy()
        
        def _radii_soft(params, beta=50.0):
            return _evaluate_params_soft(params, angles, n, beta)
        
        try:
            result = minimize(
                _radii_soft, bp_radii.copy(), method='L-BFGS-B',
                args=(40.0,), options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                       'bounds': [(0.1, 3.0)] * n}
            )
            if time_left() > 30:
                result2 = minimize(
                    _radii_soft, result.x.copy(), method='L-BFGS-B',
                    args=(200.0,), options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14,
                                           'bounds': [(0.1, 3.0)] * n}
                )
                poly = _normalize(_build_polygon_from_radii(result2.x, angles))
                if _is_strictly_convex(poly):
                    s = _polygon_min_score(poly)
                    if s > best_score:
                        best_score = s
                        best_polygon = poly
        except Exception:
            pass
    
    # Strategy 5: Coordinate descent fine-tuning
    if time_left() > 40:
        current_radii = np.sqrt(np.sum(best_polygon**2, axis=1)).copy()
        
        for outer_iter in range(20):
            if time_left() < 20:
                break
            improved = False
            base_step = 0.015 * (0.92 ** outer_iter)
            
            for i in range(n):
                if time_left() < 20:
                    break
                best_r = current_radii[i]
                best_s = _evaluate_params(current_radii, angles, n)
                for delta in [base_step, -base_step, base_step*0.5, -base_step*0.5,
                              base_step*0.25, -base_step*0.25]:
                    trial = current_radii.copy()
                    trial[i] = max(0.08, current_radii[i] + delta)
                    trial_s = _evaluate_params(trial, angles, n)
                    if trial_s < best_s:
                        best_s = trial_s
                        best_r = trial[i]
                        improved = True
                current_radii[i] = best_r
            
            if not improved:
                break
        
        final_poly = _normalize(_build_polygon_from_radii(current_radii, angles))
        if _is_strictly_convex(final_poly):
            final_score = _polygon_min_score(final_poly)
            if final_score > best_score:
                best_score = final_score
                best_polygon = final_poly
    
    # Strategy 6: Final high-precision L-BFGS-B polish
    if time_left() > 20:
        bp_radii = np.sqrt(np.sum(best_polygon**2, axis=1)).copy()
        
        def _radii_very_soft(params, beta=300.0):
            return _evaluate_params_soft(params, angles, n, beta)
        
        try:
            result = minimize(
                _radii_very_soft, bp_radii.copy(), method='L-BFGS-B',
                args=(200.0,), options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15,
                                       'bounds': [(0.1, 3.0)] * n}
            )
            poly = _normalize(_build_polygon_from_radii(result.x, angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass
    
    # Final verification
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0 or not _is_strictly_convex(best_polygon):
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon
