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
    # Clip to avoid overflow
    scaled = -beta * (scores - m)
    scaled = np.clip(scaled, -500, 500)
    return m - np.log(np.sum(np.exp(scaled))) / beta

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

def _softmin_objective_adaptive(theta_vars, beta_schedule):
    """Soft-min objective with adaptive beta based on current best score."""
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
    m = np.min(scores)
    # Adaptive beta: higher when close to optimum for more precision
    if m > 1.2:
        beta = 100.0
    elif m > 0.8:
        beta = 50.0
    else:
        beta = 20.0
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

def _golden_section_search(f, a, b, tol=1e-10, max_iter=100):
    """Golden section search for 1D minimization."""
    gr = (np.sqrt(5) - 1) / 2  # golden ratio
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    
    return (a + b) / 2

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
    # Use adaptive beta for better convergence
    try:
        result = minimize(
            _softmin_objective_adaptive, np.zeros(n), method='L-BFGS-B',
            args=(None,), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
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
    # Use adaptive beta and varying scales for exploration
    for restart in range(20):
        if time.time() > deadline - 120:
            break
        # Varying scales: mix of small, medium, and large perturbations
        if restart % 3 == 0:
            scale = 0.15
        elif restart % 3 == 1:
            scale = 0.3
        else:
            scale = 0.5
        theta_init = np.random.uniform(-scale, scale, n)
        
        try:
            result = minimize(
                _softmin_objective_adaptive, theta_init, method='L-BFGS-B',
                args=(None,), options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if time.time() > deadline - 90:
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
    
    # Strategy 4: Coordinate descent with golden-section line search for fine-tuning
    if time.time() < deadline - 45:
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
        
        for outer_iter in range(30):
            if time.time() > deadline - 30:
                break
            improved = False
            search_range = 0.05 * (0.85 ** outer_iter)
            
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                # Define 1D function for golden section search
                def obj_1d(delta):
                    trial = theta_pert.copy()
                    trial[i] = theta_pert[i] + delta
                    return _hard_objective(trial)
                
                # Golden section search in [-search_range, search_range]
                opt_delta = _golden_section_search(
                    obj_1d, -search_range, search_range, 
                    tol=1e-8, max_iter=50
                )
                
                if abs(opt_delta) > 1e-10:
                    theta_pert[i] = theta_pert[i] + opt_delta
                    improved = True
            
            if not improved:
                break
        
        final_poly = _normalize(_build_polygon_from_theta(theta_pert))
        if _is_strictly_convex(final_poly):
            final_score = _polygon_min_score(final_poly)
            if final_score > best_score:
                best_score = final_score
                best_polygon = final_poly
    
    # Strategy 5: High-precision polish with very tight tolerances
    if time.time() < deadline - 20:
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
        
        try:
            result = minimize(
                _hard_objective, theta_pert.copy(), method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 50000}
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
    
    # Final verification
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0 or not _is_strictly_convex(best_polygon):
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon
