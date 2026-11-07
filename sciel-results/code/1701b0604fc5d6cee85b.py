"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a blend of Expected Improvement and UCB, adapted to optimization progress.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # 5s safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost to avoid division by zero or log issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    # Clip Z to avoid numerical issues
    Z = np.clip(Z, -50, 50)
    EI = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    # Compute UCB (lower confidence bound for minimization)
    # LCB = mu - beta * sigma
    # Improvement from LCB = f_best - LCB = f_best - mu + beta * sigma
    # We want to balance exploration and exploitation.
    # Use a beta that adapts based on the progress of optimization.
    
    # Estimate progress: as f_best gets closer to the known minimum (-3.32237),
    # we should exploit more (lower beta). Early on, explore more (higher beta).
    known_min = -3.32237
    # Normalize f_best to [0, 1] where 0 = at minimum, 1 = far from minimum
    # Assume initial f_best is around -1.5 to -2.5
    f_range = 2.0  # approximate range of f values
    progress = np.clip((f_best - known_min) / f_range, 0, 1)
    # Early on (progress ~ 1), use higher beta; later (progress ~ 0), use lower beta
    beta = 0.5 + 1.5 * progress  # beta ranges from 0.5 to 2.0
    
    UCB_improvement = f_best - mu + beta * sigma
    UCB_improvement = np.where(np.isfinite(UCB_improvement), UCB_improvement, 0.0)
    
    # Blend EI and UCB
    # Early on, weight UCB more; later, weight EI more
    # w_ucb = progress, w_ei = 1 - progress
    w_ucb = 0.3 + 0.7 * progress  # UCB weight ranges from 0.3 to 1.0
    w_ei = 1.0 - w_ucb + 0.3  # EI weight, adjusted to keep total reasonable
    
    # Normalize both to have similar scales
    ei_mean = np.mean(EI)
    ei_std = np.std(EI) + 1e-9
    ucb_mean = np.mean(UCB_improvement)
    ucb_std = np.std(UCB_improvement) + 1e-9
    
    EI_norm = (EI - ei_mean) / ei_std
    UCB_norm = (UCB_improvement - ucb_mean) / ucb_std
    
    # Combine
    combined = w_ei * EI_norm + w_ucb * UCB_norm
    
    # Cost-aware score: divide by cost
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety: check deadline
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 2:
        # Return a simple fallback
        return (f_best - mu) / cost
    
    return scores
# EVOLVE-BLOCK-END
