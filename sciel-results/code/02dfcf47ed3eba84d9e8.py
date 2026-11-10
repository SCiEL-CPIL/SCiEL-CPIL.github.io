# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def soft_score(points, beta):
        """
        Calculate a smooth approximation of log(d_min) - log(d_max).
        Using squared distances D2.
        log(softmin(D2)) - log(softmax(D2))
        softmin(D2) = exp(-1/beta * logsumexp(-beta * D2))
        softmax(D2) = exp( 1/beta * logsumexp( beta * D2))
        log(softmin) = -1/beta * logsumexp(-beta * D2)
        log(softmax) =  1/beta * logsumexp( beta * D2)
        Objective to maximize: log(softmin) - log(softmax)
        Objective to minimize: - (log(softmin) - log(softmax))
        """
        D2 = pdist(points, 'sqeuclidean')
        
        # Avoid numerical issues with large beta * D2
        # logsumexp handles scaling internally, but let's be safe
        log_softmin = -1.0 / beta * logsumexp(-beta * D2)
        log_softmax =  1.0 / beta * logsumexp( beta * D2)
        
        # We want to maximize log_softmin - log_softmax
        # So we minimize the negative
        return -(log_softmin - log_softmax)

    def objective(params, beta):
        points = params.reshape((n, 2))
        return soft_score(points, beta)

    # Start with a 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    best_points = points.copy()
    
    # Calculate true score for tracking
    def true_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2
        
    best_score = true_score(best_points)

    # Annealing schedule for beta
    # Start with small beta (very smooth, easy to optimize) and increase to large beta (sharp, close to true objective)
    betas = [10, 50, 100, 200, 500, 1000]
    
    current_points = best_points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        # Optimize using L-BFGS with the current beta
        # We need a closure for beta
        def obj_func(p):
            return objective(p, beta)
            
        result = minimize(obj_func, current_points, method='L-BFGS-B', 
                         options={'maxiter': 2000, 'ftol': 1e-12})
        
        if result.success:
            candidate_points = result.x.reshape((n, 2))
            candidate_score = true_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
                current_points = candidate_points
            else:
                # If score didn't improve, keep previous best but update current for next iteration?
                # Usually, if soft score improved but true score didn't, it might be noise.
                # But since beta is increasing, the landscape changes.
                # Let's stick to the best true score found so far to be safe, 
                # or use the result if it's valid.
                # Actually, if the soft optimization fails to improve true score, 
                # it might be stuck in a local min of the soft function that isn't good for true.
                # Let's try to restart from best_points if it fails?
                # For now, just update current_points to result.x to continue from there,
                # but keep best_points as the best true score.
                current_points = result.x

    # Final refinement with true objective using Nelder-Mead on the best point
    if time.time() < deadline - 30:
        def true_obj(p):
            return -true_score(p.reshape((n, 2)))
            
        result = minimize(true_obj, best_points.flatten(), method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.success:
            final_score = -result.fun
            if final_score > best_score:
                best_points = result.x.reshape((n, 2))

    return best_points

# EVOLVE-BLOCK-END
