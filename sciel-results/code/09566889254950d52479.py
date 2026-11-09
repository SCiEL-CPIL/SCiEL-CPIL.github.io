"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track cumulative time spent in acquisition calls
_acq_start_time = [time.time()]
_acq_cumulative_time = [0.0]
DEADLINE_SECONDS = 310.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP posterior mean at candidate points
    sigma : ndarray of shape (2000,)
        GP posterior standard deviation at candidate points
    f_best : float
        Lowest observed objective value so far
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _acq_start_time[0] + _acq_cumulative_time[0]
    if elapsed > DEADLINE_SECONDS:
        # Return a simple deterministic score to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    call_start = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-aware Expected Improvement with uncertainty penalty
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Compute standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # The EI has two terms:
    # 1. (f_best - mu) * Phi(z): improvement term (higher when mu < f_best)
    # 2. sigma * phi(z): uncertainty term (higher when sigma is large)
    #
    # In early sparse data, sigma can be large everywhere, causing the uncertainty
    # term to dominate and push exploration to high-noise regions. We want to
    # penalize the uncertainty term more aggressively in the early regime.
    
    # Strategy: Use a modified EI where the uncertainty term is scaled down
    # based on how uncertain the improvement is. We use a "confidence-weighted"
    # approach where we weight the EI by the probability that the point actually
    # improves upon f_best, but with a stricter threshold.
    
    # Alternative: Use the "Expected Improvement with penalty" (EIP)
    # EIP = EI - kappa * sigma
    # This directly penalizes high uncertainty.
    
    # Let's use an adaptive kappa that depends on f_best:
    # When f_best is high (early), we want more exploration but not too much
    # When f_best is low (late), we want exploitation
    
    # Clip f_best to [0, 1] for stable interpolation
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # Adaptive kappa: smaller when f_best is good (exploit), larger when poor (explore)
    # But we want to prevent the uncertainty term from dominating, so we cap it
    kappa_min = 0.1
    kappa_max = 0.5
    kappa = kappa_min + (kappa_max - kappa_min) * f_best_clipped
    
    # Compute EIP: EI - kappa * sigma
    eip = ei - kappa * sigma
    
    # Also compute a UCB-based score for blending
    # UCB for minimization: -mu + beta * sigma
    beta_min = 0.3
    beta_max = 1.0
    beta = beta_min + (beta_max - beta_min) * f_best_clipped
    ucb = -mu + beta * sigma
    
    # Blend EIP and UCB: use more EIP when exploiting, more UCB when exploring
    # But keep the blend stable to avoid oscillation
    weight_eip = 0.6 + 0.4 * (1.0 - f_best_clipped)  # 0.6 to 1.0
    weight_eip = np.clip(weight_eip, 0.5, 0.9)
    weight_ucb = 1.0 - weight_eip
    
    scores = weight_eip * eip + weight_ucb * ucb
    
    # Divide by cost to be cost-aware
    scores = scores / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Update cumulative time
    call_elapsed = time.time() - call_start
    _acq_cumulative_time[0] += call_elapsed
    
    return scores
# EVOLVE-BLOCK-END
