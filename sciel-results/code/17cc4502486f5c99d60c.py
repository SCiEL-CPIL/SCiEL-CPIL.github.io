"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Pareto-frontier filtering and UCB.
    
    Strategy:
    1. Filter out candidates where mu is significantly worse than f_best.
       This restricts the search to regions where improvement is plausible.
    2. Apply a pure Upper Confidence Bound (UCB) score to the filtered candidates:
       score = -mu + beta * sigma
       This aggressively exploits high-uncertainty areas while still favoring
       lower predicted means among the filtered set.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (minimum) objective value observed so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate
        more promising candidates to evaluate next.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not np.isfinite(f_best):
        return np.random.randn(len(mu))
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Pareto-frontier filtering
    # Mask out candidates whose predicted mean is worse than f_best by more than a tolerance.
    # We use a tolerance of 0.0 (strictly better or equal) but to be safe against noise,
    # we might allow a very small tolerance. However, the prompt says "worse than f_best 
    # by more than a small tolerance". Let's use a tolerance of 0.0 for strict filtering 
    # or a small epsilon. Given the Weierstrass function is continuous, 
    # restricting to mu <= f_best is a strong filter.
    
    # To be robust, let's allow a small tolerance. If mu is much worse, it's unlikely to improve.
    # Let's set tolerance to 0.0 initially. If too few points pass, we might need to relax.
    # But the strategy says "mask out any candidate whose predicted mean mu is worse than 
    # the current best f_best by more than a small tolerance".
    
    # Let's use a tolerance that scales with the spread of mu or a fixed small value.
    # A fixed small value like 1e-3 or 0.01? 
    # Given the domain is [-0.5, 0.5] and function values can be small, 
    # let's use a tolerance of 0.01.
    
    tolerance = 0.01
    mask = mu <= (f_best + tolerance)
    
    # If no points pass the filter (which can happen if GP is overconfident),
    # fall back to selecting points with the lowest mu (most promising) or all points.
    if not np.any(mask):
        # Select the best 10% of points based on mu
        threshold = np.percentile(mu, 10)
        mask = mu <= threshold
    
    # Step 2: Apply UCB to the filtered subset
    # UCB = -mu + beta * sigma
    # Higher score means more promising.
    # We want to maximize -mu + beta * sigma.
    
    # Choose beta. A common value is 2.0 or 2.5.
    # Since we are filtering to regions near f_best, the mu values are close to f_best.
    # The sigma term becomes more important for exploration within this region.
    beta = 2.0
    
    scores = np.zeros_like(mu)
    
    # Compute UCB only for the masked points
    # For unmasked points, score remains 0 (or very low) so they won't be selected.
    # To be safe, set unmasked scores to a very low value.
    scores[~mask] = -1e9
    
    # For masked points:
    scores[mask] = -mu[mask] + beta * sigma_safe[mask]
    
    # Check time
    if time.time() > deadline:
        # Return best result found so far (which is just the computed scores)
        pass
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores

# EVOLVE-BLOCK-END
