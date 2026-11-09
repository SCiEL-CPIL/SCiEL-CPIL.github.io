import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys


def _vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex in [0, 1].
    """
    d = np.sort(distances)
    
    if np.any(d < 0.05):
        return 0.0
        
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score


def _vertex_score_smooth(distances, beta=20.0):
    """
    Smooth approximation of _vertex_score using log-sum-exp for min.
    Returns a smooth proxy in [0, 1].
    """
    d = np.sort(distances)
    
    # Smooth penalty for d < 0.05
    pen1 = np.exp(-beta * (0.05 - d))  # large when d < 0.05
    if np.max(pen1) > 1e6:
        return 0.0
    
    diffs = d[1:] - d[:-1]
    k = int(np.argmin(diffs))
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score


def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    
    # Check for repeated points
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
        return False
        
    # Check turns
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(np.abs(cross) < 1e-8):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    return True


def _normalize(points):
    """Normalize points: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return None
        
    return points_norm / mean_dist


def _polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score after normalization.
    """
    points_norm = _normalize(points)
    if points_norm is None:
        return 0.0
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    min_score = 1.0
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = _vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score


def _polygon_score_smooth(points, beta=20.0):
    """
    Smooth polygon score using log-sum-exp for the min over vertices.
    """
    points_norm = _normalize(points)
    if points_norm is None:
        return 0.0
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    scores = np.zeros(n)
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        scores[i] = _vertex_score_smooth(dists, beta)
    
    # Smooth min via log-sum-exp
    min_score = np.min(scores)
    # Use soft-min: -log-sum-exp(-scores) / beta
    neg_scores = -scores
    max_neg = np.max(neg_scores)
    if max_neg > 1e6:
        return 0.0
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min = -log_sum_exp / beta
    
    return max(0.0, min(1.0, soft_min))


def _polygon_from_angles(angle_offsets):
    """
    angle_offsets: array of length 20.
    Returns (20, 2) array with unit radii and given angle offsets from regular polygon.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    radii = np.ones(n)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _polygon_from_angles_radii(angle_offsets, radii):
    """
    angle_offsets: array of length 20.
    radii: array of length 20.
    Returns (20, 2) array.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _score_from_angles(angle_offsets, smooth=False, beta=20.0):
    """Score from angle offsets only (unit radii)."""
    points = _polygon_from_angles(angle_offsets)
    if smooth:
        return _polygon_score_smooth(points, beta)
    return _polygon_score(points)


def _score_from_angles_radii(params, smooth=False, beta=20.0):
    """Score from combined angle_offsets + radii params (length 40)."""
    n = 20
    angle_offsets = params[:n]
    radii = params[n:]
    points = _polygon_from_angles_radii(angle_offsets, radii)
    if smooth:
        return _polygon_score_smooth(points, beta)
    return _polygon_score(points)


def _bipartite_polygon(ratio=1.0):
    """
    Create a bipartite-like polygon: alternate between two radii.
    This often gives good initial geometry for this problem.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    # Alternate radii
    for i in range(n):
        if i % 2 == 0:
            radii[i] = 1.0
        else:
            radii[i] = ratio
    points = _polygon_from_angles_radii(np.zeros(n), radii)
    return points


def _score_bipartite(ratio, smooth=False, beta=20.0):
    """Score for bipartite polygon with given ratio."""
    points = _bipartite_polygon(ratio)
    if smooth:
        return _polygon_score_smooth(points, beta)
    return _polygon_score(points)


def _objective_angles(params, smooth=False, beta=20.0):
    score = _score_from_angles(params, smooth=smooth, beta=beta)
    if score == 0.0:
        return 100.0
    return -score


def _objective_angles_radii(params, smooth=False, beta=20.0):
    score = _score_from_angles_radii(params, smooth=smooth, beta=beta)
    if score == 0.0:
        return 100.0
    return -score


def _local_refinement_angles(angle_offsets, max_iter=200, step_size=0.02, smooth=True, beta=20.0):
    """
    Local refinement on angle offsets only.
    """
    n = 20
    current = angle_offsets.copy()
    current_score = _score_from_angles(current, smooth=smooth, beta=beta)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for delta in [-step_size, step_size]:
                trial = current.copy()
                trial[i] += delta
                trial_score = _score_from_angles(trial, smooth=smooth, beta=beta)
                if trial_score > current_score + 1e-8:
                    current = trial
                    current_score = trial_score
                    improved = True
        if not improved:
            break
        step_size *= 0.85
    
    return current, current_score


def _local_refinement_radii(radii, max_iter=200, step_size=0.02, smooth=True, beta=20.0):
    """
    Local refinement on radii only (angles fixed to regular).
    """
    n = 20
    current = radii.copy()
    current_score = _score_from_angles_radii(np.concatenate([np.zeros(n), current]), smooth=smooth, beta=beta)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for delta in [-step_size, step_size]:
                trial = current.copy()
                trial[i] = max(0.1, trial[i] + delta)
                trial_score = _score_from_angles_radii(np.concatenate([np.zeros(n), trial]), smooth=smooth, beta=beta)
                if trial_score > current_score + 1e-8:
                    current = trial
                    current_score = trial_score
                    improved = True
        if not improved:
            break
        step_size *= 0.85
    
    return current, current_score


def _local_search(points, max_iter=100, step_size=0.01):
    """
    Perform a local search around the given points.
    """
    n = 20
    current_points = points.copy()
    current_score = _polygon_score(current_points)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    trial = current_points.copy()
                    trial[i] += [dx, dy]
                    trial_score = _polygon_score(trial)
                    if trial_score > current_score:
                        current_points = trial
                        current_score = trial_score
                        improved = True
        if not improved:
            break
        step_size *= 0.9
    
    return current_points, current_score


def _extract_params(points):
    """Extract angle_offsets and radii from points."""
    n = 20
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    radii = np.linalg.norm(points_centered, axis=1)
    angles = np.arctan2(points_centered[:, 1], points_centered[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    
    sort_idx = np.argsort(angles)
    radii = radii[sort_idx]
    angles = angles[sort_idx]
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angle_offsets = angles - base_angles
    for i in range(n):
        if angle_offsets[i] > np.pi:
            angle_offsets[i] -= 2 * np.pi
        elif angle_offsets[i] < -np.pi:
            angle_offsets[i] += 2 * np.pi
    
    return angle_offsets, radii


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum vertex score.
    
    Uses multi-phase approach:
    1. Bipartite initialization
    2. DE on angles (smooth surrogate)
    3. DE on radii (smooth surrogate)  
    4. Nelder-Mead on angles (smooth)
    5. Nelder-Mead on angles+radii (exact)
    6. Local search polish
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    n = 20
    
    # Phase 0: Bipartite initialization
    best_points = _bipartite_polygon(1.0)
    best_score = _polygon_score(best_points)
    
    # Try a few bipartite ratios
    for ratio in [0.8, 0.9, 1.1, 1.2]:
        if time.time() > deadline - 500.0:
            break
        pts = _bipartite_polygon(ratio)
        s = _polygon_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Phase 1: DE on angles with smooth surrogate
    angle_bound = 0.3 * (2 * np.pi / n)
    angle_bounds = [(-angle_bound, angle_bound)] * n
    
    try:
        res_de_ang = differential_evolution(
            lambda p: _objective_angles(p, smooth=True, beta=15.0),
            angle_bounds,
            maxiter=40,
            popsize=10,
            tol=1e-5,
            seed=42,
            polish=False
        )
        if res_de_ang.fun < 100.0:
            cur_score = _score_from_angles(res_de_ang.x)
            if cur_score > best_score:
                best_score = cur_score
                best_points = _polygon_from_angles(res_de_ang.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 200.0:
        return best_points
    
    # Phase 2: DE on radii with smooth surrogate (angles fixed to best)
    _, best_radii = _extract_params(best_points)
    _, best_angle_offsets = _extract_params(best_points)
    
    radius_bounds = (0.5, 1.5)
    radius_bounds_list = [radius_bounds] * n
    
    def _obj_radii(params):
        return _objective_angles_radii(np.concatenate([best_angle_offsets, params]), smooth=True, beta=15.0)
    
    try:
        res_de_rad = differential_evolution(
            _obj_radii,
            radius_bounds_list,
            maxiter=30,
            popsize=8,
            tol=1e-5,
            seed=123,
            polish=False
        )
        if res_de_rad.fun < 100.0:
            cur_score = _score_from_angles_radii(np.concatenate([best_angle_offsets, res_de_rad.x]))
            if cur_score > best_score:
                best_score = cur_score
                best_points = _polygon_from_angles_radii(best_angle_offsets, res_de_rad.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 150.0:
        return best_points
    
    # Phase 3: Nelder-Mead on angles (smooth)
    _, best_angle_offsets = _extract_params(best_points)
    
    try:
        res_nm_ang = minimize(
            lambda p: _objective_angles(p, smooth=True, beta=20.0),
            best_angle_offsets,
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000, 'adaptive': True}
        )
        if res_nm_ang.fun < 100.0:
            cur_score = _score_from_angles(res_nm_ang.x)
            if cur_score > best_score:
                best_score = cur_score
                best_points = _polygon_from_angles(res_nm_ang.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 100.0:
        return best_points
    
    # Phase 4: Nelder-Mead on angles+radii (exact)
    best_angle_offsets, best_radii = _extract_params(best_points)
    full_params = np.concatenate([best_angle_offsets, best_radii])
    
    for restart in range(3):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            init_params = full_params.copy()
        else:
            init_params = full_params + 0.01 * np.random.randn(2 * n)
        
        try:
            res_nm_full = minimize(
                lambda p: _objective_angles_radii(p, smooth=False),
                init_params,
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': 3000, 'maxfev': 8000, 'adaptive': True}
            )
            if res_nm_full.fun < 100.0:
                cur_score = _score_from_angles_radii(res_nm_full.x)
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = _polygon_from_angles_radii(res_nm_full.x[:n], res_nm_full.x[n:])
                    best_angle_offsets, best_radii = _extract_params(best_points)
                    full_params = np.concatenate([best_angle_offsets, best_radii])
        except Exception:
            continue
    
    # Phase 5: Local search polish
    if time.time() < deadline - 15.0:
        polished_points, polished_score = _local_search(best_points, max_iter=80, step_size=0.005)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    return best_points


if __name__ == "__main__":
    # Test
    pts = equidistant_points_polygon20()
    print("Score:", _polygon_score(pts))
