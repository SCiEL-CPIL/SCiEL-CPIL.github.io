"""Rewritten program for the 50-equal-circles-in-a-circle packing task."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _pairwise_indices(n):
    """Return indices for all pairs i < j."""
    i_idx, j_idx = np.triu_indices(n, 1)
    return i_idx, j_idx

def _score_vec(points):
    """Compute the packing radius for given centers (vectorized)."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    i_idx, j_idx = _pairwise_indices(n)
    pair_dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=1) / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _smooth_min_objective(params, n, i_idx, j_idx, beta=100.0):
    """
    Smooth surrogate for the max-min packing problem.
    We minimize the negative of a soft-min of all constraint distances.
    """
    pts = params.reshape(-1, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1) / 2.0
    all_dists = np.concatenate([boundary, dists])
    
    # Soft min via log-sum-exp
    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    smooth_min = -log_sum / beta
    return -smooth_min

def _smooth_min_grad(params, n, i_idx, j_idx, beta=100.0):
    """
    Gradient of the smooth min objective.
    """
    pts = params.reshape(-1, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1) / 2.0
    all_dists = np.concatenate([boundary, dists])
    
    # Soft min via log-sum-exp
    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    exp_terms = np.exp(neg_dists - max_neg)
    sum_exp = np.sum(exp_terms)
    
    # d(-smooth_min)/d(all_dists) = -d(smooth_min)/d(all_dists)
    # smooth_min = -log(sum(exp(-all_dists))) / beta
    # d(smooth_min)/d(all_dists) = (sum(exp(-all_dists) * 1)) / (beta * sum(exp(-all_dists)))
    # = exp_terms / (beta * sum_exp)
    # So d(-smooth_min)/d(all_dists) = -exp_terms / (beta * sum_exp)
    
    weights = -exp_terms / (beta * sum_exp)
    
    grad = np.zeros_like(pts)
    
    # Boundary terms: boundary[i] = 1 - |pts[i]|
    # d(boundary[i])/d(pts[i]) = -pts[i] / |pts[i]|
    for i in range(n):
        if norms[i] > 1e-10:
            grad[i] += weights[i] * (-pts[i] / norms[i])
    
    # Pairwise terms: dists[k] = |pts[i]-pts[j]| / 2
    # d(dists[k])/d(pts[i]) = (pts[i]-pts[j]) / (2 * |pts[i]-pts[j]|)
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = np.linalg.norm(diffs[k])
        if d > 1e-10:
            factor = weights[n + k] / (2.0 * d)
            grad[i] += factor * diffs[k]
            grad[j] -= factor * diffs[k]
    
    return grad.flatten()

def _polish_smooth(points, n, deadline, maxiter=500, beta=100.0):
    """Polish using smooth min objective with gradient-based optimization."""
    i_idx, j_idx = _pairwise_indices(n)
    x0 = points.flatten()
    
    def objective(x):
        return _smooth_min_objective(x, n, i_idx, j_idx, beta)
    
    def jac(x):
        return _smooth_min_grad(x, n, i_idx, j_idx, beta)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 20})
        if np.all(np.isfinite(res.x)):
            cand = res.x.reshape(-1, 2)
            return cand
    except Exception:
        pass
    return points

def _rigid_ring_params_to_points(ring_radii, ring_angles, n_ring_counts):
    """
    Convert rigid ring parameters to 50 points.
    ring_radii: array of ring radii (length = num_rings)
    ring_angles: array of ring angular offsets (length = num_rings)
    n_ring_counts: list of number of points per ring
    """
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    for ring_idx in range(len(ring_radii)):
        r = ring_radii[ring_idx]
        offset = ring_angles[ring_idx]
        n_pts = n_ring_counts[ring_idx]
        angles = offset + np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            points.append([x, y])
    
    return np.array(points[:50])

def _rigid_ring_score(params, n_ring_counts):
    """Score for rigid ring configuration."""
    ring_radii = params[:len(n_ring_counts)]
    ring_angles = params[len(n_ring_counts):2*len(n_ring_counts)]
    points = _rigid_ring_params_to_points(ring_radii, ring_angles, n_ring_counts)
    return _score_vec(points)

def _optimize_rigid_rings(deadline):
    """
    Optimize rigid ring configuration using differential evolution.
    Known optimal topology: 1 center, rings of 6, 12, 18, 13.
    """
    n_ring_counts = [6, 12, 18, 13]
    num_rings = len(n_ring_counts)
    
    # Bounds for ring radii and angles
    bounds = []
    for i in range(num_rings):
        # Ring radii: from small to near boundary
        bounds.append((0.1, 0.9))
    for i in range(num_rings):
        # Angular offsets: 0 to 2*pi
        bounds.append((0.0, 2 * np.pi))
    
    # Objective: negative score (we want to maximize score)
    def objective(params):
        return -_rigid_ring_score(params, n_ring_counts)
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=200, popsize=30, tol=1e-10,
            seed=42, polish=False,
            init='sobol',
            update_interval=1,
            mutation=(0.3, 0.8),
            recombination=0.8
        )
        if np.all(np.isfinite(result.x)):
            ring_radii = result.x[:num_rings]
            ring_angles = result.x[num_rings:2*num_rings]
            points = _rigid_ring_params_to_points(ring_radii, ring_angles, n_ring_counts)
            return points
    except Exception:
        pass
    
    # Fallback: reasonable initial guess
    ring_radii = np.array([0.25, 0.45, 0.65, 0.85])
    ring_angles = np.zeros(4)
    points = _rigid_ring_params_to_points(ring_radii, ring_angles, n_ring_counts)
    return points

def _free_optimize(points, n, deadline):
    """Free optimization of all 100 coordinates using smooth min."""
    i_idx, j_idx = _pairwise_indices(n)
    x0 = points.flatten()
    
    def objective(x):
        return _smooth_min_objective(x, n, i_idx, j_idx, beta=100.0)
    
    def jac(x):
        return _smooth_min_grad(x, n, i_idx, j_idx, beta=100.0)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30})
        if np.all(np.isfinite(res.x)):
            cand = res.x.reshape(-1, 2)
            return cand
    except Exception:
        pass
    return points

def _high_precision_polish(points, n, deadline):
    """
    High-precision polish with increasing beta for sharper min approximation.
    """
    i_idx, j_idx = _pairwise_indices(n)
    best_points = points.copy()
    best_score = _score_vec(points)
    
    for beta in [50.0, 100.0, 200.0, 400.0, 800.0]:
        if time.time() > deadline - 2.0:
            break
        
        x0 = best_points.flatten()
        
        def objective(x):
            return _smooth_min_objective(x, n, i_idx, j_idx, beta)
        
        def jac(x):
            return _smooth_min_grad(x, n, i_idx, j_idx, beta)
        
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 20})
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(-1, 2)
                score = _score_vec(cand)
                if score > best_score:
                    best_score = score
                    best_points = cand.copy()
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = N
    best_points = None
    best_score = 0.0
    
    # Phase 1: Rigid ring optimization using differential evolution
    if time.time() < deadline - 60.0:
        ring_points = _optimize_rigid_rings(deadline - 60.0)
        ring_score = _score_vec(ring_points)
        best_points = ring_points.copy()
        best_score = ring_score
    
    # Phase 2: Free optimization from best rigid ring configuration
    if time.time() < deadline - 30.0 and best_points is not None:
        free_points = _free_optimize(best_points, n, deadline - 30.0)
        free_score = _score_vec(free_points)
        if free_score > best_score:
            best_score = free_score
            best_points = free_points.copy()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5.0 and best_points is not None:
        polished, pol_score = _high_precision_polish(best_points, n, deadline - 5.0)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()
    
    # Phase 4: Multi-start refinement
    if time.time() < deadline - 10.0 and best_points is not None:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            pert = best_points + np.random.randn(n, 2) * 0.005
            free_points = _free_optimize(pert, n, deadline - 10.0)
            free_score = _score_vec(free_points)
            if free_score > best_score:
                best_score = free_score
                best_points = free_points.copy()
            
            if time.time() < deadline - 5.0:
                polished, pol_score = _high_precision_polish(best_points, n, deadline - 5.0)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished.copy()
    
    # Final validity check
    if best_points is None:
        # Fallback: simple hexagonal
        points = []
        s = 0.2
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            best_points = points[idx]
        else:
            best_points = np.random.randn(n, 2) * 0.5
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, 2))
    
    return best_points
# EVOLVE-BLOCK-END
