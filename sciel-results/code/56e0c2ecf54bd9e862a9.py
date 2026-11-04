import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area computation
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    # Soft minimum via log-sum-exp
    max_area = np.max(areas)
    if max_area < 1e-12:
        return 0.0
    shifted = areas - max_area
    log_sum = max_area + np.log(np.sum(np.exp(shifted / temperature)))
    soft_min = max_area - temperature * np.log(np.sum(np.exp(shifted / temperature)))

    return soft_min / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    for r0 in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        for phi0 in [0, np.pi/n_in if n_in > 0 else 0, np.pi/(2*n_in) if n_in > 0 else 0, np.pi/(n_out if n_out > 0 else 1)]:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    return best_score, best_params

def _polish_points(points, time_limit=5.0):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    start_time = time.time()
    
    def neg_score_flat(x):
        p = x.reshape(N, 2)
        return -_calculate_score(p)
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # First L-BFGS-B pass
    try:
        res = minimize(neg_score_flat, points.flatten(), method='L-BFGS-B',
                      options={'maxiter':300, 'ftol':1e-14, 'gtol':1e-12, 'maxls':50})
        new_pts = res.x.reshape(N, 2)
        new_score = _calculate_score(new_pts)
        if new_score > best_score:
            best_pts = new_pts
            best_score = new_score
    except Exception:
        pass
    
    # Second L-BFGS-B pass if time allows
    if time.time() - start_time < time_limit * 0.5:
        try:
            res2 = minimize(neg_score_flat, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-16, 'gtol':1e-14, 'maxls':30})
            pts2 = res2.reshape(N, 2)
            score2 = _calculate_score(pts2)
            if score2 > best_score:
                best_pts = pts2
                best_score = score2
        except Exception:
            pass
    
    # Coordinate descent polish on worst triangles
    if time.time() - start_time < time_limit * 0.7:
        best_pts = _coordinate_descent_polish(best_pts, time_limit - (time.time() - start_time))
        best_score = _calculate_score(best_pts)
    
    return best_pts, best_score

def _coordinate_descent_polish(points, time_limit=2.0):
    """Iteratively adjust each point to improve the minimum triangle area."""
    start_time = time.time()
    pts = points.copy()
    score = _calculate_score(pts)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = pts.copy()
                        trial[i] += [dx, dy]
                        trial_score = _calculate_score(trial)
                        if trial_score > score:
                            pts = trial
                            score = trial_score
                            improved = True
        # Halve step size if no improvement
        if not improved:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    splits = [
        (14, 0),
        (13, 1),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (8, 6),
        (7, 7)
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0)
            score = _calculate_score(pts)
            # Polish the regular polygon with generous time
            polished_pts, polished_score = _polish_points(pts, time_limit=4.0)
            try_update(polished_score, polished_pts)
            continue
        
        score, params = _optimize_two_ring(n_out, n_in, time_limit=10.0)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            # Polish with full coordinate optimization
            remaining = deadline - time.time() - 25
            polish_time = max(3.0, min(8.0, remaining))
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)

    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=remaining)
        try_update(final_score, final_pts)

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        # Try to polish even the fallback
        remaining = deadline - time.time() - 3
        polished_pts, polished_score = _polish_points(best_points, time_limit=max(3.0, remaining))
        best_points = polished_pts
        best_score = polished_score

    return best_points
