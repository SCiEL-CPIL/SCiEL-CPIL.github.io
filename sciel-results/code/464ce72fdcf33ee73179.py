import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _soft_min(x, p):
    """
    Compute the soft minimum of an array x using log-sum-exp.
    approx min(x) ~ -1/p * log(sum(exp(-p*x)))
    """
    # For numerical stability, subtract the minimum
    x_min = np.min(x)
    # If x is very large, exp(-p*x) underflows, but that's fine, it becomes 0.
    # We want to avoid overflow in exp.
    # log(sum(exp(-p*(x - x_min)))) - p*x_min
    # = log(sum(exp(-p*x + p*x_min))) - p*x_min
    # = log(exp(p*x_min) * sum(exp(-p*x))) - p*x_min
    # = p*x_min + log(sum(exp(-p*x))) - p*x_min
    # = log(sum(exp(-p*x)))
    # Wait, the formula for soft min is:
    # m = -1/p * log(sum(exp(-p*x_i)))
    # To stabilize:
    # m = -1/p * log(sum(exp(-p*(x_i - x_min)))) - x_min
    #   = -1/p * log(sum(exp(-p*x_i + p*x_min))) - x_min
    #   = -1/p * (p*x_min + log(sum(exp(-p*x_i)))) - x_min
    #   = -x_min - 1/p * log(sum(exp(-p*x_i))) + x_min ... no.
    
    # Let's just use:
    # val = -1/p * log(sum(exp(-p*x)))
    # If x has large positive values, exp(-p*x) is small.
    # If x has large negative values, exp(-p*x) is large -> overflow.
    # But distances are positive.
    # So x >= 0.
    # exp(-p*x) <= 1. No overflow.
    # Underflow is possible for large x, but that just makes the term 0, which is correct for min.
    
    ex = np.exp(-p * x)
    s = np.sum(ex)
    if s == 0:
        return np.inf
    return -np.log(s) / p

def _get_distances(pts):
    """
    Returns a flattened array of all pairwise distances and boundary distances.
    """
    n = pts.shape[0]
    # Pairwise distances
    # Use broadcasting for vectorization
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only take upper triangle to avoid duplicates and zeros
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    # dist to wall = 0.5 - abs(coord)
    # We want min over all coords and all points
    # b_dists = 0.5 - np.abs(pts)
    # We take the min of all these values.
    # To feed into soft_min, we can just flatten them.
    b_dists = 0.5 - np.abs(pts)
    
    return np.concatenate([pair_dists, b_dists.flatten()])

def _objective_de(x):
    """
    Objective for Differential Evolution (minimization).
    We want to maximize the minimum distance, so we minimize a potential energy
    that is low when points are far apart.
    U = sum(1/d^2) for pairs + penalty for boundaries.
    """
    pts = x.reshape(N, 3)
    
    # Boundary penalty
    # If any point is outside [ -0.5, 0.5 ], penalize heavily.
    # Actually, DE bounds will keep them inside if we set bounds correctly.
    # But let's add a small penalty just in case or to encourage centering.
    # The bounds in DE will be [-0.5, 0.5].
    
    # Pairwise repulsion
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Avoid division by zero
    pair_dists = np.maximum(pair_dists, 1e-6)
    
    # Potential: sum of inverse squares
    # This drives points apart.
    U = np.sum(1.0 / (pair_dists ** 2))
    
    return U

def _objective_lbgfgs(x, p=100.0):
    """
    Objective for L-BFGS-B (minimization).
    Minimizes -soft_min(distances).
    Equivalent to maximizing soft_min(distances).
    """
    pts = x.reshape(N, 3)
    dists = _get_distances(pts)
    
    # We want to maximize the min distance.
    # soft_min approximates min.
    # We minimize -soft_min.
    return -_soft_min(dists, p)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Global Optimization with Differential Evolution
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # DE parameters
    # maxiter=30, popsize=10
    # This might take a bit, so we check time.
    
    try:
        de_result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=20,
            popsize=10,
            tol=0.01,
            seed=42,
            polish=False, # We do our own polish
            init='sobol'
        )
        best_x = de_result.x
    except Exception:
        # Fallback to random if DE fails or times out unexpectedly
        best_x = np.random.uniform(-0.5, 0.5, N * 3)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline * 0.6:
        # If DE took too long, skip to a quick polish or just return best so far
        # But DE is usually fast enough for 150 vars.
        pass

    # 2. Local Optimization with L-BFGS-B
    # We use the best point from DE as the initial guess.
    # We minimize -soft_min(distances).
    
    # We can run L-BFGS-B multiple times with increasing p to sharpen the min.
    # Or just one run with a large p.
    
    current_x = best_x.copy()
    
    # Run L-BFGS-B
    # maxiter=200
    # ftol, gtol
    try:
        res = minimize(
            _objective_lbgfgs,
            current_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success or res.x is not None:
            current_x = res.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline * 0.8:
        # Return current best
        pts = current_x.reshape(N, 3)
        return pts

    # 3. Optional: Second polish pass with higher p
    # If we have time, run again with p=200
    if time.time() - start_time < deadline * 0.9:
        try:
            res2 = minimize(
                lambda x: _objective_lbgfgs(x, p=200.0),
                current_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.success or res2.x is not None:
                current_x = res2.x
        except Exception:
            pass

    pts = current_x.reshape(N, 3)
    
    # Final sanity check: ensure points are within bounds
    # L-BFGS-B should handle this, but just in case
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts

# EVOLVE-BLOCK-END
