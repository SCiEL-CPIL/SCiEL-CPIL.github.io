"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    Uses a Linear Cost-Aware UCB strategy.

    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate

    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple normalized improvement
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize inputs
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Linear UCB components
    # Exploitation: How much better is the mean than the current best?
    # We want to maximize (f_best - mu), so we use -mu as the core signal relative to f_best.
    # However, standard UCB for minimization is: -mu + beta * sigma.
    # Here we scale it to be cost-aware.
    
    # Exploration: Uncertainty
    # We need to balance exploration (sigma) against cost.
    
    # Normalization for stability
    # The magnitude of mu and sigma depends on the GP and the function scale.
    # Rosenbrock values can be large.
    # Let's normalize sigma by its max to keep weights interpretable, 
    # but actually, raw sigma is often better if beta is tuned.
    # Let's look at the scale of f_best.
    
    # A robust approach:
    # Score = (f_best - mu) / cost + beta * sigma / cost
    # This is equivalent to (f_best - mu + beta * sigma) / cost
    
    # Let's use a dynamic beta based on the remaining budget or just a fixed one.
    # Given the "Linear Cost-Aware UCB" inspiration which scored 1.724,
    # we assume a form like:
    # score = (improvement + beta * uncertainty) / cost
    
    # Improvement term: f_best - mu
    # Uncertainty term: sigma
    
    # We need to choose beta.
    # If beta is too high, we explore too much.
    # If beta is too low, we exploit too much.
    # Standard UCB often uses beta ~ log(t). Here we don't have t directly, 
    # but we have cost.
    
    # Let's try a fixed beta that balances the scales.
    # Since mu and sigma are in objective units, and cost is in [1, 2] roughly,
    # we want the exploration term to be significant but not dominant initially.
    
    # Let's estimate a scale for beta.
    # If f_best is large, sigma might be large.
    # Let's use a beta that is relative to the current uncertainty range.
    
    # Simple fixed beta first, as suggested by the linear UCB success.
    # The inspiration mentioned "Linear Cost-Aware UCB".
    # Let's implement: score = (f_best - mu + beta * sigma) / cost
    
    # What value for beta?
    # In many BO tasks, beta is around 1-3 for standardized GP, but here GP is standardized targets?
    # "standardized targets" means mu and sigma are in original units? 
    # "mu and sigma (a standard deviation) are in objective units."
    # So sigma is in the same units as f_best.
    
    # Let's try beta = 2.0. This is a common starting point for UCB.
    # We can also make it adaptive: beta = 2.0 * (sigma.mean() / (abs(f_best) + 1e-6)) 
    # But fixed is simpler and less prone to numerical issues.
    
    beta = 2.0
    
    # Calculate the UCB value for minimization
    # We want to find points where mu is low (good) and sigma is high (uncertain).
    # Objective to maximize: (f_best - mu) + beta * sigma
    # Note: (f_best - mu) is the expected improvement in mean.
    
    ucb_value = (f_best - mu) + beta * sigma
    
    # Cost-aware: Divide by cost
    # Higher cost reduces the score.
    score = ucb_value / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Add a small exploration bonus for very high sigma regions if UCB is too conservative?
    # No, stick to the linear UCB structure as it showed promise.
    
    return score
# EVOLVE-BLOCK-END
