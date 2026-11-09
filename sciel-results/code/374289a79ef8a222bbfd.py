import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 60.0

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _objective_de(x_flat):
    """
    Objective function for Differential Evolution.
    Minimizes negative score.
    """
    points = x_flat.reshape(14, 3)
    score = _raw_score(points)
    return -score

def _objective_grad(x_flat):
    """
    Objective function for L-BFGS-B polishing.
    Uses a smoothed approximation of the min/max to help gradients,
    or just relies on the raw value if smoothing is too complex.
    For L-BFGS-B, we can use the raw value, but it might stop early due to non-differentiability.
    A common trick is to use a soft-min/soft-max.
    However, given the small N (14), the raw value often works if we provide a good start.
    Let's stick to the raw value but ensure we handle the gradient carefully if we were providing one.
    Since L-BFGS-B estimates gradients if none are provided (or we can provide them), 
    let's provide the analytical subgradient for the active pairs.
    """
    points = x_flat.reshape(14, 3)
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10 or d_min < 1e-10:
        return 1.0, np.zeros_like(x_flat)
        
    score = (d_min / d_max) ** 2
    val = -score
    
    # Calculate gradient
    # We need d(score)/d(points).
    # score = (d_min/d_max)^2
    # Let r = d_min/d_max. score = r^2.
    # d(score) = 2r * dr
    # dr = (d_max * dd_min - d_min * dd_max) / d_max^2
    # dd_min is the gradient of the minimum distance.
    # dd_max is the gradient of the maximum distance.
    
    # To get gradients of min and max, we need to know which pairs are active.
    # For min: all pairs with dist == d_min (within eps).
    # For max: all pairs with dist == d_max (within eps).
    
    # This is complex to implement robustly for every step in L-BFGS-B without significant overhead.
    # Instead, we will use L-BFGS-B with finite difference gradients (default) or just run it 
    # and hope it converges. Actually, providing a gradient is much faster.
    
    # Let's implement a simple subgradient.
    # We need the pairwise differences.
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    D2 = np.maximum(D2, 1e-30)
    dists_mat = np.sqrt(D2)
    
    # Identify active min pairs
    eps = 1e-5
    min_pairs = np.argwhere(dists_mat < d_min + eps)
    max_pairs = np.argwhere(dists_mat > d_max - eps)
    
    # Filter to i < j to avoid double counting and self-loops
    min_pairs = [(i, j) for i, j in min_pairs if i < j]
    max_pairs = [(i, j) for i, j in max_pairs if i < j]
    
    grad = np.zeros((14, 3))
    
    if len(min_pairs) > 0:
        # Average gradient of the min distance
        # d(dist_ij)/d(X_i) = (X_i - X_j) / dist_ij
        # d(dist_ij)/d(X_j) = (X_j - X_i) / dist_ij
        g_min = np.zeros((14, 3))
        for i, j in min_pairs:
            d_vec = points[i] - points[j]
            d_val = np.linalg.norm(d_vec)
            if d_val > 1e-10:
                unit_vec = d_vec / d_val
                g_min[i] += unit_vec
                g_min[j] -= unit_vec
        g_min /= len(min_pairs)
        
    else:
        g_min = np.zeros((14, 3))
        
    if len(max_pairs) > 0:
        g_max = np.zeros((14, 3))
        for i, j in max_pairs:
            d_vec = points[i] - points[j]
            d_val = np.linalg.norm(d_vec)
            if d_val > 1e-10:
                unit_vec = d_vec / d_val
                g_max[i] += unit_vec
                g_max[j] -= unit_vec
        g_max /= len(max_pairs)
    else:
        g_max = np.zeros((14, 3))
        
    # score = (d_min/d_max)^2
    # dscore/dX = 2 * (d_min/d_max) * (1/d_max * dd_min/dX - d_min/d_max^2 * dd_max/dX)
    # dscore/dX = 2 * d_min / d_max^2 * dd_min/dX - 2 * d_min^2 / d_max^3 * dd_max/dX
    
    coeff_min = 2.0 * d_min / (d_max * d_max)
    coeff_max = 2.0 * (d_min * d_min) / (d_max * d_max * d_max)
    
    grad = coeff_min * g_min - coeff_max * g_max
    
    # We are minimizing -score, so gradient is -grad_score
    grad_flat = (-grad).flatten()
    
    return val, grad_flat

def _polish(points0, deadline):
    """
    Refine the points using L-BFGS-B.
    """
    x0 = points0.flatten()
    
    # Check deadline
    if time.time() > deadline:
        return points0
    
    try:
        res = minimize(
            lambda x: _objective_grad(x)[0],
            x0,
            jac=lambda x: _objective_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        cand = res.x.reshape(14, 3)
        return cand
    except Exception:
        return points0

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    
    best_score = 0.0
    best_points = np.random.default_rng(0).standard_normal((14, 3))
    
    # Define bounds for DE. 
    # The problem is scale invariant. We can constrain the bounding box.
    # A reasonable box is [-1, 1] in each dimension.
    bounds = [(-1.0, 1.0)] * 42
    
    # Run Differential Evolution
    # maxiter and popsize control the effort.
    # We will run it for as long as we have time, or until it converges.
    
    # DE is stochastic. We can run multiple iterations if time permits, 
    # but usually one long run is better.
    
    # To use time effectively, we can update the best result periodically.
    # However, DE's callback is tricky to use for early stopping in a single call.
    # Instead, we will run DE with a generous maxiter, and rely on the deadline.
    # If DE takes too long, we just return the best found so far.
    
    # A better strategy: Run DE with a moderate number of iterations, then polish.
    # If time remains, run DE again with a different seed or more iterations?
    # DE is global, so one good run is usually sufficient.
    
    # Let's try to run DE with maxiter=100, popsize=15.
    # This is roughly 1500 function evaluations. Each is fast (pdist of 14 points).
    # 1500 evals should take < 1 second.
    # We can afford many more.
    
    # Let's estimate time per eval.
    # pdist(14,3) is very fast.
    # Let's aim for ~50,000 to 100,000 evaluations in total.
    
    # We will loop, running DE with increasing maxiter, or just running DE once with high maxiter.
    # Since DE is deterministic given the seed, we can just run it once with high maxiter.
    # But if it gets stuck, we might want to restart.
    
    # Strategy:
    # 1. Run DE with maxiter=200.
    # 2. Polish the result.
    # 3. If time remains, run DE again from the polished result? No, DE is global.
    #    Maybe run DE with different seeds?
    
    # Let's run DE with a high maxiter and see how long it takes.
    # If it finishes quickly, we can run more seeds.
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > deadline - 10: # Leave time for polishing
            break
            
        # Run DE
        # tol=1e-8, maxiter=100
        try:
            res = differential_evolution(
                _objective_de,
                bounds,
                seed=seed,
                maxiter=50, # Keep it moderate, DE is slow per iter
                popsize=15,
                tol=1e-6,
                polish=False # We will polish manually to control time
            )
            
            cand_points = res.x.reshape(14, 3)
            
            # Polish
            polished_points = _polish(cand_points, deadline)
            
            score = _raw_score(polished_points)
            if score > best_score:
                best_score = score
                best_points = polished_points
                
        except Exception:
            pass
            
    # If we still have time, do a final polish on the best points found
    if time.time() < deadline - 5:
        polished_best = _polish(best_points, deadline)
        score = _raw_score(polished_best)
        if score > best_score:
            best_score = score
            best_points = polished_best

    # Ensure valid output
    if best_points.shape != (14, 3) or not np.all(np.isfinite(best_points)):
        # Fallback to a simple configuration
        # 14 points on a sphere?
        # Just return random if something went wrong
        best_points = np.random.default_rng(123).standard_normal((14, 3))
        
    return best_points

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
