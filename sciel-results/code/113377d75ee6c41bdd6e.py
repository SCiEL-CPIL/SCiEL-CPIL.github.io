# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist

def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be +-2*pi
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True

def _polygon_score_fast(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    # Compute all pairwise distances at once
    diffs = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diffs, diffs)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    scores = np.empty(n)
    for i in range(n):
        dists_i = np.sort(dists[i])
        
        if dists_i[0] < 0.05:
            scores[i] = 0.0
            continue
        
        gaps = dists_i[1:] - dists_i[:-1]
        k = np.argmin(gaps)
        D = (dists_i[k] + dists_i[k + 1]) / 2
        
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        if k == 0:
            four_dists = dists_i[0:4]
        elif k == 17:
            four_dists = dists_i[15:19]
        else:
            four_dists = dists_i[k - 1:k + 3]
        
        ratios_a = four_dists / D
        ratios_b = D / four_dists
        raw = np.mean(np.maximum(ratios_a, ratios_b))
        scores[i] = min(1.0, 1.0 / raw)
    
    return np.min(scores)

def _smooth_polygon_score(polygon, beta=10.0):
    """Compute a smooth approximation of the minimum vertex score using soft-min."""
    n = len(polygon)
    diffs = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diffs, diffs)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    scores = np.empty(n)
    for i in range(n):
        dists_i = np.sort(dists[i])
        
        if dists_i[0] < 0.05:
            scores[i] = 0.0
            continue
        
        gaps = dists_i[1:] - dists_i[:-1]
        k = np.argmin(gaps)
        D = (dists_i[k] + dists_i[k + 1]) / 2
        
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        if k == 0:
            four_dists = dists_i[0:4]
        elif k == 17:
            four_dists = dists_i[15:19]
        else:
            four_dists = dists_i[k - 1:k + 3]
        
        ratios_a = four_dists / D
        ratios_b = D / four_dists
        raw = np.mean(np.maximum(ratios_a, ratios_b))
        scores[i] = min(1.0, 1.0 / raw)
    
    # Soft-min: -log(sum(exp(-beta*s)))/beta
    max_s = np.max(scores)
    if max_s <= 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (scores - max_s))))
    soft_min = max_s - np.log(np.sum(np.exp(-beta * scores))) / beta
    return soft_min

def _score_from_radii(radii):
    """Compute score given radii for fixed angles."""
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _score_from_radii_smooth(radii, beta=10.0):
    """Compute smooth score given radii for fixed angles."""
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _smooth_polygon_score(polygon, beta)

def _score_from_angles_radii(params):
    """Compute score given both angles and radii (20 angles + 20 radii = 40 params)."""
    n = 20
    angles = params[:n]
    radii = params[n:]
    
    angles = np.sort(angles)
    radii = np.maximum(radii, 0.1)
    
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _score_from_angles_radii_smooth(params, beta=10.0):
    """Compute smooth score given both angles and radii."""
    n = 20
    angles = params[:n]
    radii = params[n:]
    
    angles = np.sort(angles)
    radii = np.maximum(radii, 0.1)
    
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _smooth_polygon_score(polygon, beta)

def _local_refinement(params, step_size, n_steps, deadline, score_func):
    """Perform local refinement with adaptive step sizes."""
    n = len(params)
    best_params = params.copy()
    best_score = score_func(best_params)
    if best_score < 0:
        return params, 0.0
    
    current_step = step_size
    for step in range(n_steps):
        if time.time() > deadline - 15.0:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 15.0:
                break
            
            # Try positive and negative perturbations
            for direction in [1, -1]:
                trial = best_params.copy()
                trial[i] += direction * current_step
                # Ensure angles are in valid range
                if i < 20:
                    trial[i] = np.clip(trial[i], 0.01, 2*np.pi - 0.01)
                else:
                    trial[i] = np.maximum(trial[i], 0.3)
                
                score = score_func(trial)
                if score > best_score:
                    best_params = trial
                    best_score = score
                    improved = True
        
        if not improved:
            current_step *= 0.5
            if current_step < 1e-6:
                break
        else:
            current_step *= 0.85
    
    return best_params, best_score

def _nm_optimize(params, deadline, score_func, bounds=None):
    """Nelder-Mead optimization."""
    n = len(params)
    
    def objective(x):
        if time.time() > deadline - 15.0:
            return 1e9
        score = score_func(x)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, params, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return params, score_func(params)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses multi-phase optimization over both angles and radii.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Phase 1: Start with regular polygon
    best_radii = np.ones(n)
    best_score = _score_from_radii(best_radii)
    
    # Phase 2: Differential evolution on radii (global search)
    if time.time() < deadline - 180.0:
        def de_objective(radii):
            if time.time() > deadline - 180.0:
                return 1e9
            score = _score_from_radii(radii)
            if score < 0:
                return 1e9
            return -score
        
        bounds = [(0.3, 2.0)] * n
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=80, 
                                               popsize=20, tol=1e-6, seed=42,
                                               polish=False, workers=1)
            if de_result.fun < 1e8:
                de_radii = de_result.x
                de_score = _score_from_radii(de_radii)
                if de_score > best_score:
                    best_radii = de_radii
                    best_score = de_score
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on radii (smooth surrogate first, then exact)
    if time.time() < deadline - 120.0:
        # Use smooth surrogate for faster convergence
        nm_radii_smooth, nm_score_smooth = _nm_optimize(best_radii, deadline, 
                                                         lambda r: _score_from_radii_smooth(r, 15.0), None)
        if nm_score_smooth > 0:
            nm_radii_exact, nm_score_exact = _nm_optimize(nm_radii_smooth, deadline, 
                                                           _score_from_radii, None)
            if nm_score_exact > best_score:
                best_radii = nm_radii_exact
                best_score = nm_score_exact
    
    # Phase 4: Local refinement on radii
    if time.time() < deadline - 60.0:
        refined_radii, refined_score = _local_refinement(best_radii, 0.01, 100, deadline, _score_from_radii)
        if refined_score > best_score:
            best_radii = refined_radii
            best_score = refined_score
    
    # Phase 5: Joint angle+radii optimization using smooth surrogate
    if time.time() < deadline - 90.0:
        initial_params = np.concatenate([angles, best_radii])
        
        def joint_smooth_func(params):
            if time.time() > deadline - 90.0:
                return -1.0
            return _score_from_angles_radii_smooth(params, 15.0)
        
        # NM with smooth surrogate
        nm_joint_smooth, nm_joint_score_smooth = _nm_optimize(initial_params, deadline, 
                                                               joint_smooth_func, None)
        if nm_joint_score_smooth > 0:
            # Refine with exact score
            nm_joint_exact, nm_joint_score_exact = _nm_optimize(nm_joint_smooth, deadline, 
                                                                _score_from_angles_radii, None)
            if nm_joint_score_exact > best_score:
                # Extract polygon from joint params
                a = nm_joint_exact[:20]
                r = nm_joint_exact[20:]
                polygon = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
                best_polygon = _normalize(polygon)
                if _is_valid_convex(best_polygon):
                    best_score = nm_joint_score_exact
                    return best_polygon
    
    # Phase 6: DE on joint angles+radii (if time permits)
    if time.time() < deadline - 60.0:
        def joint_de_objective(params):
            if time.time() > deadline - 60.0:
                return 1e9
            score = _score_from_angles_radii(params)
            if score < 0:
                return 1e9
            return -score
        
        bounds = [(0.01, 2*np.pi - 0.01)] * 20 + [(0.3, 2.0)] * 20
        try:
            de_joint = differential_evolution(joint_de_objective, bounds, maxiter=30,
                                              popsize=15, tol=1e-5, seed=123,
                                              polish=False, workers=1)
            if de_joint.fun < 1e8:
                de_score = _score_from_angles_radii(de_joint.x)
                if de_score > best_score:
                    a = de_joint.x[:20]
                    r = de_joint.x[20:]
                    polygon = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
                    best_polygon = _normalize(polygon)
                    if _is_valid_convex(best_polygon):
                        best_score = de_score
                        return best_polygon
        except Exception:
            pass
    
    # Phase 7: Final high-precision polish
    if time.time() < deadline - 30.0:
        # Reconstruct from best radii with regular angles
        p = np.stack([best_radii * np.cos(angles), best_radii * np.sin(angles)], axis=1)
        best_polygon = _normalize(p)
        
        # Final NM polish with very tight tolerance
        def final_objective(radii):
            if time.time() > deadline - 30.0:
                return 1e9
            radii = np.maximum(radii, 0.1)
            polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
            polygon = _normalize(polygon)
            if not _is_valid_convex(polygon):
                return 1e9
            return -_polygon_score_fast(polygon)
        
        try:
            res = minimize(final_objective, best_radii, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 1e8:
                final_radii = np.maximum(res.x, 0.1)
                final_polygon = np.stack([final_radii * np.cos(angles), final_radii * np.sin(angles)], axis=1)
                final_polygon = _normalize(final_polygon)
                if _is_valid_convex(final_polygon):
                    final_score = _polygon_score_fast(final_polygon)
                    if final_score > best_score:
                        return final_polygon
        except Exception:
            pass
    
    # Reconstruct final polygon
    best_radii = np.maximum(best_radii, 0.1)
    p = np.stack([best_radii * np.cos(angles), best_radii * np.sin(angles)], axis=1)
    best_polygon = _normalize(p)
    
    # Final validity check
    if not _is_valid_convex(best_polygon):
        # Fallback to regular polygon
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(p)
    
    return best_polygon

# EVOLVE-BLOCK-END
