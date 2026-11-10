# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Objective for minimization: negative of the score."""
        pts = x.reshape(n, 2)
        return -get_score(pts)

    # Strategy: Differential Evolution on the exact non-smooth objective,
    # followed by local refinement with L-BFGS-B on a soft-min surrogate.

    # Define bounds for the search space.
    # We can normalize the scale, so let's search in a reasonable box.
    # Since the problem is scale-invariant (ratio), we can fix the scale
    # by constraining the diameter or just searching in a box and letting
    # the optimizer figure it out. However, DE needs bounds.
    # A good range is roughly [-10, 10] for coordinates if the optimal
    # configuration has diameter ~1 or so.
    # Let's use a slightly wider range to be safe.
    bounds = [(-10.0, 10.0)] * (n * 2)

    # Run Differential Evolution
    # maxiter and popsize control the budget.
    # 16 points * 2 dims = 32 variables.
    # popsize=15 -> 480 individuals.
    # maxiter=500 -> 240,000 evaluations.
    # Each evaluation is pdist (120 distances) + min/max. Very fast.
    # This should finish well within the time limit.
    
    best_points = None
    best_score = -np.inf

    # Check time before starting
    if time.time() < deadline - 10:
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=1000,
                popsize=20,
                tol=1e-10,
                polish=True,  # Let DE do a local polish at the end
                init='sobol',
                workers=1
            )
            best_points = result.x.reshape(n, 2)
            best_score = -result.fun
        except Exception:
            pass

    # Local refinement using a smooth surrogate (soft-min)
    # to potentially escape shallow local minima or polish the solution.
    # We use a soft-min of distances as the objective to maximize.
    # Maximize soft_min(dists) / soft_max(dists) or similar.
    # Actually, maximizing d_min/d_max is equivalent to maximizing log(d_min) - log(d_max).
    # Soft approx:
    # log(d_min) ~ -t * log(sum(exp(-d_i/t)))
    # log(d_max) ~ t * log(sum(exp(d_i/t)))
    # So we want to maximize: -t*log(sum(exp(-d_i/t))) - t*log(sum(exp(d_i/t)))
    # Or minimize the negative of that.

    def soft_objective(x, t=10.0):
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        
        # Soft min
        a_min = -dists / t
        a_min_max = np.max(a_min)
        soft_min = -t * (a_min_max + np.log(np.sum(np.exp(a_min - a_min_max)) / len(dists)))
        
        # Soft max
        a_max = dists / t
        a_max_max = np.max(a_max)
        soft_max = t * (a_max_max + np.log(np.sum(np.exp(a_max - a_max_max)) / len(dists)))
        
        # We want to maximize soft_min - soft_max (approx log(d_min) - log(d_max))
        # So minimize -(soft_min - soft_max)
        return -(soft_min - soft_max)

    # Refine the best point found by DE
    if best_points is not None:
        current_best = best_points
        current_score = best_score
        
        # Try a few local optimizations with different temperatures
        for t in [10.0, 5.0, 1.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    soft_objective,
                    current_best.flatten(),
                    args=(t,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                refined_points = res.x.reshape(n, 2)
                refined_score = get_score(refined_points)
                
                if refined_score > current_score:
                    current_best = refined_points
                    current_score = refined_score
            except Exception:
                pass

        best_points = current_best
        best_score = current_score

    # Fallback if DE failed or timed out
    if best_points is None:
        # Use a heuristic initialization: regular grid or hexagonal lattice
        # Let's try a 4x4 grid
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        best_score = get_score(best_points)

    # Final check
    if time.time() > deadline - 1:
        return best_points

    return best_points


# EVOLVE-BLOCK-END
