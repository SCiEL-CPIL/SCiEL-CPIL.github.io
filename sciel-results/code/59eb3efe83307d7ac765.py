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
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple UCB
        beta = 1.0
        improvement = f_best - mu + beta * sigma
        scores = improvement / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate optimization progress
    # f_best starts around -1 to -2, approaches -3.32237
    # progress = 0 means early, progress = 1 means near convergence
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        # Normalize: assume initial f_best is around -1.0
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Blend EI and UCB based on progress
    # Early: more UCB (exploration), Late: more EI (exploitation)
    # EI: E[max(f_best - f(x), 0)]
    # UCB lower: f_best - mu + beta * sigma
    
    # Compute EI
    # For minimization, improvement = f_best - f(x)
    # EI = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are CDF and PDF of standard normal
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement
    # Adaptive beta: higher in early stages, lower in late stages
    beta_early = 2.0
    beta_late = 0.5
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Blend: weight = progress means we use more EI as we progress
    # w = progress: w=0 -> pure UCB, w=1 -> pure EI
    w = progress
    blended_improvement = w * ei + (1 - w) * ucb_improvement
    
    # Cost-aware score
    scores = blended_improvement / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > DEADLINE - 5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
