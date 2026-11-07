import numpy as np
import time
from scipy.special import erfc, log_ndtr, ndtr

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function for Bayesian optimization.
    
    Uses LCB = mu - beta * sigma, where beta is dynamically adjusted.
    The score is (f_best - LCB) / cost, which prioritizes points with high predicted
    improvement relative to their evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.clip(cost, 1e-10, None)
    
    # Dynamic exploration coefficient:
    # Start with a moderate exploration level and adapt based on the spread of predictions.
    # We want to balance exploration (uncertainty) and exploitation (predicted improvement).
    
    # Compute a measure of exploration vs exploitation
    # When sigma is large relative to the range of mu, we should explore more.
    mu_range = np.max(mu) - np.min(mu)
    sigma_mean = np.mean(sigma)
    
    # Dynamic beta: starts around 1.5-2.0, adjusts based on the problem state
    # If predictions are uncertain (high sigma relative to mu range), explore more
    if mu_range > 1e-10:
        uncertainty_ratio = sigma_mean / (mu_range + 1e-10)
        # Map uncertainty ratio to beta in [1.0, 3.0]
        beta = 1.0 + 2.0 * min(uncertainty_ratio, 1.0)
    else:
        beta = 2.0
    
    # LCB: lower confidence bound (for minimization, we want low values)
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # The improvement relative to f_best is f_best - LCB
    # If LCB < f_best, this is positive (expected improvement)
    improvement = f_best - lcb
    
    # Clip to non-negative (LCB can be > f_best in some regions)
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-normalized score
    scores = improvement / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple regret reduction per cost with exploration
        simple_scores = np.maximum(f_best - mu, 0.0) / cost_safe
        # Add a small exploration bonus based on sigma
        exploration_bonus = 0.5 * sigma_safe / cost_safe
        simple_scores = simple_scores + exploration_bonus
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
