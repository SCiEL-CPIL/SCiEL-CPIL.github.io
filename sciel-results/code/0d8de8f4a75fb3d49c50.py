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

def _softmin(scores, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(scores)
    if m < 0:
        return m
    return m - np.log(np.sum(np.exp(-beta * (scores - m)))) / beta

def _softmin_objective(theta_vars, beta=50.0):
    """Smooth negative min vertex score using angular perturbations."""
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles + theta_vars
    
    x = np.cos(thetas)
    y = np.sin(thetas)
    polygon = np.stack([x, y], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_strictly_convex(polygon):
        return 10.0
    
    scores = _compute_all_scores(polygon)
    return -_softmin(scores, beta)

def _hard_objective(theta_vars):
    """Hard negative min vertex score using angular perturbations."""
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles + theta_vars
    
    x = np.cos(thetas)
    y = np.sin(thetas)
    polygon = np.stack([x, y], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_strictly_convex(polygon):
        return 10.0
    
    return -_polygon_min_score(polygon)

def _build_polygon_from_theta(theta_vars):
    """Build polygon from angular perturbations."""
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles + theta_vars
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.stack([x, y], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum per-vertex equidistance score.
    Uses angular perturbation with log-sum-exp smooth surrogate for L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    n = 20
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_polygon = _normalize(best_polygon)
    best_score = _polygon_min_score(best_polygon)
    
    np.random.seed(42)
    
    # Strategy 1: L-BFGS-B with log-sum-exp soft-min from zero perturbation
    try:
        result = minimize(
            _softmin_objective, np.zeros(n), method='L-BFGS-B',
            args=(50.0,), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if time.time() < deadline - 120:
            poly = _normalize(_build_polygon_from_theta(result.x))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
    except Exception:
        pass
    
    # Strategy 2: Multiple random restarts with L-BFGS-B soft-min
    for restart in range(15):
        if time.time() > deadline - 90:
            break
        scale = 0.3 / (1 + restart * 0.5)
        theta_init = np.random.uniform(-scale, scale, n)
        
        try:
            result = minimize(
                _softmin_objective, theta_init, method='L-BFGS-B',
                args=(50.0,), options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if time.time() > deadline - 60:
                break
            poly = _normalize(_build_polygon_from_theta(result.x))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass
    
    # Strategy 3: Hard objective polish with Nelder-Mead from best soft-min result
    if time.time() < deadline - 60:
        current_theta = np.zeros(n)
        # Try to extract from best polygon
        bp = best_polygon
        centroid = np.mean(bp, axis=0)
        bp_centered = bp - centroid
        current_theta = np.arctan2(bp_centered[:, 1], bp_centered[:, 0])
        # Convert to perturbations relative to base angles
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        current_theta = current_theta - base_angles
        # Unwrap
        for i in range(1, n):
            if current_theta[i] - current_theta[i-1] > np.pi:
                current_theta[i] -= 2*np.pi
            elif current_theta[i] - current_theta[i-1] < -np.pi:
                current_theta[i] += 2*np.pi
        
        try:
            result = minimize(
                _hard_objective, current_theta.copy(), method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < -best_score:
                poly = _normalize(_build_polygon_from_theta(result.x))
                if _is_strictly_convex(poly):
                    s = _polygon_min_score(poly)
                    if s > best_score:
                        best_score = s
                        best_polygon = poly
        except Exception:
            pass
    
    # Strategy 4: Coordinate descent on angles for fine-tuning
    if time.time() < deadline - 30:
        bp = best_polygon
        centroid = np.mean(bp, axis=0)
        bp_centered = bp - centroid
        theta_current = np.arctan2(bp_centered[:, 1], bp_centered[:, 0])
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        theta_pert = theta_current - base_angles
        for i in range(1, n):
            if theta_pert[i] - theta_pert[i-1] > np.pi:
                theta_pert[i] -= 2*np.pi
            elif theta_pert[i] - theta_pert[i-1] < -np.pi:
                theta_pert[i] += 2*np.pi
        
        for outer_iter in range(20):
            if time.time() > deadline - 15:
                break
            improved = False
            base_step = 0.01 * (0.9 ** outer_iter)
            
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                best_theta = theta_pert[i]
                best_s = _hard_objective(theta_pert)
                for delta in [base_step, -base_step, base_step*0.5, -base_step*0.5]:
                    trial = theta_pert.copy()
                    trial[i] = theta_pert[i] + delta
                    trial_s = _hard_objective(trial)
                    if trial_s < best_s:
                        best_s = trial_s
                        best_theta = trial[i]
                        improved = True
                theta_pert[i] = best_theta
            
            if not improved:
                break
        
        final_poly = _normalize(_build_polygon_from_theta(theta_pert))
        if _is_strictly_convex(final_poly):
            final_score = _polygon_min_score(final_poly)
            if final_score > best_score:
                best_score = final_score
                best_polygon = final_poly
    
    # Final verification
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0 or not _is_strictly_convex(best_polygon):
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon
