# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin of 20 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(points_flat):
        """
        Objective function for minimization (we want to maximize score).
        Returns -score.
        """
        points = points_flat.reshape(16, 2)
        score = compute_score(points)
        if not np.isfinite(score):
            return 1e10
        return -score

    best_points = None
    best_score = -np.inf

    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        points = np.array(points, dtype=float)
        if points.shape != (16, 2) or not np.all(np.isfinite(points)):
            return
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    n = 16
    d = 2

    # Define bounds for differential evolution.
    # We expect the optimal configuration to be within a reasonable range.
    # Since the objective is scale-invariant (ratio of distances), we can constrain the points to a box.
    # A box of [-1, 1] x [-1, 1] is a good starting guess, but to be safe, let's use [-2, 2].
    bounds = [(-2, 2)] * (n * d)

    # Run Differential Evolution
    # DE is a global optimizer that uses a population of candidate solutions.
    # It is robust to local minima and does not require gradients.
    # We use 'best1bin' mutation strategy, which is standard.
    # maxiter is set high to allow thorough exploration, but we monitor time.
    
    # We will run DE in chunks or just run it and hope it finishes, but DE can be slow.
    # Let's estimate: 32 variables, popsize=15 (default is 15*len(bounds) = 480, which is too big).
    # Let's use a smaller popsize to speed up, e.g., 20.
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,  # We will do our own polishing
            init='sobol'   # Sobol sequence for better initial distribution
        )
        
        if de_result.success or de_result.fun < -best_score:
            update_best(de_result.x.reshape(16, 2))
            
    except Exception as e:
        # If DE fails or times out, we might not have a good result yet.
        # But usually it returns something.
        pass

    # Check time
    if time.time() > deadline - 30:
        if best_points is not None:
            return best_points
        # Fallback if nothing found
        return np.random.randn(16, 2)

    # Now, polish the best result using Nelder-Mead on the hard objective.
    # Nelder-Mead is a local optimizer that works well for non-smooth functions.
    # We start from the best point found by DE.
    
    if best_points is not None:
        x0 = best_points.flatten()
        
        try:
            nm_result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'fatol': 1e-12,
                    'xatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if nm_result.fun < -best_score:
                update_best(nm_result.x.reshape(16, 2))
                
        except Exception:
            pass

    # If we still have time, try a few random restarts with Nelder-Mead
    # to see if we can find a better local minimum.
    if time.time() < deadline - 60:
        for seed in [1, 2, 3, 42, 123, 456, 789]:
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed)
            # Initialize in a smaller range to encourage convergence to a valid packing
            random_points = np.random.uniform(-1, 1, (16, 2))
            
            try:
                nm_result = minimize(
                    objective,
                    random_points.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'fatol': 1e-10,
                        'xatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if nm_result.fun < -best_score:
                    update_best(nm_result.x.reshape(16, 2))
                    
            except Exception:
                pass

    # Final check
    if best_points is None:
        # Fallback to a simple grid if nothing else worked
        best_points = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T

    return best_points

# EVOLVE-BLOCK-END
