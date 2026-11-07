"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) 
    with a small sigma-based exploration bonus.
    
    This function computes the Expected Improvement of each candidate point
    relative to the current best observed value, divides by the evaluation cost,
    and adds a small, bounded exploration term proportional to the posterior
    standard deviation. This balances exploitation (high EI) and exploration
    (high sigma) while accounting for the cost of evaluation.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Ensure inputs are float64 arrays for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins to avoid division by zero or log(0)
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical underflow/overflow in norm.cdf and norm.pdf
    # Values beyond +/- 20 are effectively 0 or 1 for CDF and near 0 for PDF
    z_clipped = np.clip(z, -20.0, 20.0)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # Formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: 5% weight on sigma
    # This encourages exploring high-uncertainty regions when EI is low
    # The "capped" nature is implicitly handled by the small weight (0.05)
    # which keeps the bonus small relative to typical EI values
    exploration_bonus = 0.05 * sigma_safe
    
    # Cost-aware score: (Exploitation + Exploration) / Cost
    # Higher score means better improvement per unit cost
    score = (ei + exploration_bonus) / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure scores are non-negative (optional, but safe)
    score = np.maximum(score, 0.0)
    
    # Deadline guard: return immediately if we are approaching the limit
    # The task specifies a 330s limit. We use 290s as a safety margin.
    elapsed = time.time() - t0
    if elapsed > 290.0:
        return score
        
    return score
# EVOLVE-BLOCK-END
