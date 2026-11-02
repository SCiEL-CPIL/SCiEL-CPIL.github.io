import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TRIPLES = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]

# Precompute indices for fast triangle area computation
TRI_I = np.array([t[0] for t in TRIPLES])
TRI_J = np.array([t[1] for t in TRIPLES])
TRI_K = np.array([t[2] for t in TRIPLES])


def _convex_hull_area(points):
    """Compute convex hull area using scipy.spatial.ConvexHull."""
    try:
        # Check for collinearity or insufficient points
        if np.allclose(points[:, 0], points[:, 0][0]) or np.allclose(points[:, 1], points[:, 1][0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: sort by angle and use shoelace
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        sorted_idx = np.argsort(angles)
        p = points[sorted_idx]
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        return 0.5 * abs(np.sum(x * yn - xn * y))


def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    pi = points[TRI_I]
    pj = points[TRI_J]
    pk = points[TRI_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    return 0.5 * cross


def _smooth_min(areas, beta):
    """
    Compute the smooth minimum (soft-min) of the areas.
    Uses the log-sum-exp trick for stability.
    soft_min(A) = -1/beta * log(sum(exp(-beta * A)))
    """
    # Shift for numerical stability
    min_val = np.min(areas)
    shifted = -beta * (areas - min_val)
    # Clip to avoid overflow in exp, though shifted values are <= 0
    # exp(shifted) will be in (0, 1]
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return min_val - log_sum_exp / beta


def _score_smooth(points, beta):
    """
    Compute the smooth score: soft_min(triangle_areas) / hull_area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    areas = _triangle_areas(points)
    soft_min_area = _smooth_min(areas, beta)
    
    return soft_min_area / hull_area


def _neg_score_smooth(x_flat, beta):
    """Objective for minimization: negative smooth score."""
    points = x_flat.reshape(N, 2)
    return -_score_smooth(points, beta)


def _make_regular_polygon(n, radius=1.0, phase=0.0):
    """Create a regular n-gon."""
    theta = np.linspace(phase, 2 * np.pi + phase, n, endpoint=False)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def _make_concentric(n_outer, n_inner, r_inner, phase_outer=0.0, phase_inner=0.0):
    """Create points on two concentric regular polygons."""
    pts = []
    if n_outer > 0:
        pts.append(_make_regular_polygon(n_outer, 1.0, phase_outer))
    if n_inner > 0:
        pts.append(_make_regular_polygon(n_inner, r_inner, phase_inner))
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)


def _optimize_from_start(x0, deadline, max_iter=500, beta_start=10.0, beta_end=1000.0, n_anneal_steps=10):
    """
    Run L-BFGS-B optimization with annealing of the soft-min parameter beta.
    """
    if time.time() > deadline:
        return None, -np.inf
    
    best_x = x0.copy()
    best_score = _score_smooth(best_x.reshape(N, 2), beta_start)
    
    # Anneal beta from low to high
    betas = np.linspace(beta_start, beta_end, n_anneal_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        stop_flag = [False]
        
        def callback(xk):
            if time.time() > deadline:
                stop_flag[0] = True
                return True
            return False
        
        try:
            # Use L-BFGS-B with bounds to keep points in a reasonable range
            res = opt.minimize(
                lambda x: _neg_score_smooth(x, beta),
                best_x,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10},
                callback=callback
            )
            
            if not stop_flag[0]:
                current_x = res.x
                current_score = _score_smooth(current_x.reshape(N, 2), beta)
                
                # We compare scores using the *current* beta for consistency in the optimization step,
                # but for the final result, we care about the true min (beta -> inf).
                # However, for selecting the best path, we can track the score at the current beta.
                # To be safe, we also evaluate the "hard" score periodically.
                
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()
                    
        except Exception:
            pass
            
    # Final evaluation with high beta (approximating true min)
    final_points = best_x.reshape(N, 2)
    final_score = _score_smooth(final_points, beta_end)
    
    return final_points, final_score


def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 130s to allow for return overhead.
    deadline = start_time + 130.0
    
    best_score = -np.inf
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(_make_regular_polygon(N, 1.0, 0.0))
    
    # 2. Concentric configurations
    # 7 outer, 6 inner
    initial_configs.append(_make_concentric(7, 6, 0.5, 0.0, 0.0))
    initial_configs.append(_make_concentric(7, 6, 0.5, 0.0, np.pi / 7))
    initial_configs.append(_make_concentric(7, 6, 0.6, 0.0, np.pi / 7))
    initial_configs.append(_make_concentric(7, 6, 0.4, 0.0, np.pi / 6))
    
    # 8 outer, 5 inner
    initial_configs.append(_make_concentric(8, 5, 0.5, 0.0, 0.0))
    initial_configs.append(_make_concentric(8, 5, 0.5, 0.0, np.pi / 8))
    
    # 6 outer, 7 inner
    initial_configs.append(_make_concentric(6, 7, 0.5, 0.0, 0.0))
    initial_configs.append(_make_concentric(6, 7, 0.5, 0.0, np.pi / 6))
    
    # 9 outer, 4 inner
    initial_configs.append(_make_concentric(9, 4, 0.5, 0.0, 0.0))
    initial_configs.append(_make_concentric(9, 4, 0.5, 0.0, np.pi / 9))
    
    # 3. Random perturbations of regular polygon
    np.random.seed(42)
    for _ in range(5):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        r = np.ones(N) + np.random.uniform(-0.1, 0.1, N)
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initial_configs.append(pts)
        
    # 4. Slightly irregular polygons (varying radii)
    for _ in range(5):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.ones(N) + np.random.uniform(-0.2, 0.2, N)
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initial_configs.append(pts)
        
    # Evaluate initial configs to find good starting points
    # We use a moderate beta for initial scoring to avoid noise
    beta_eval = 50.0
    config_scores = []
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        s = _score_smooth(config, beta_eval)
        config_scores.append((s, idx))
        if s > best_score:
            best_score = s
            best_points = config.copy()
            
    # Sort configs by score
    config_scores.sort(reverse=True)
    
    # Optimize from the top configurations
    n_tried = 0
    max_starts = 10
    for s_init, idx in config_scores:
        if time.time() > deadline - 20:
            break
        if n_tried >= max_starts:
            break
            
        x0 = initial_configs[idx].flatten()
        
        # Adjust max_iter based on remaining time
        time_left = deadline - time.time()
        if time_left < 10:
            break
            
        pts, s = _optimize_from_start(x0, deadline, max_iter=300)
        n_tried += 1
        
        if pts is not None:
            # Evaluate with high beta to get a more accurate "hard" score
            s_hard = _score_smooth(pts, 1000.0)
            if s_hard > best_score:
                best_score = s_hard
                best_points = pts
    
    # Final polish: If we have a good candidate, try Nelder-Mead on the hard objective?
    # Nelder-Mead is derivative-free and can handle non-smoothness, but it's slow.
    # Given the smooth objective worked well, we might just do one more L-BFGS with high beta.
    if best_points is not None and time.time() < deadline - 5:
        # Final refinement with high beta
        x0 = best_points.flatten()
        try:
            res = opt.minimize(
                lambda x: _neg_score_smooth(x, 1000.0),
                x0,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts_final = res.x.reshape(N, 2)
                s_final = _score_smooth(pts_final, 1000.0)
                if s_final > best_score:
                    best_score = s_final
                    best_points = pts_final
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _make_regular_polygon(N, 1.0, 0.0)
        
    return best_points
