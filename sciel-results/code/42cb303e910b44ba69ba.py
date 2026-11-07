"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 320.0  # safety margin before 330s kill
_START_TIME = time.time()


def _norm_cdf(x):
    """Numerically stable standard normal CDF."""
    return 0.5 * (1.0 + np.vectorize(__import__('scipy').special.erf)(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balances expected improvement against evaluation cost,
    with an exploration bonus for high-uncertainty regions.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP standard deviations in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    global _DEADLINE, _START_TIME
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Return a safe fallback: just pick the lowest cost point
        return np.ones_like(cost) / cost
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    cost_safe = np.maximum(cost, 1e-8)
    cost_aware_ei = ei / cost_safe
    
    # Exploration bonus: encourage visiting high-uncertainty areas,
    # weighted inversely by cost to avoid expensive exploratory moves
    exploration = sigma_safe / cost_safe
    
    # Combine: primary signal is cost-aware EI, with a modest exploration term
    # The exploration coefficient is tuned to not overwhelm EI in converged regions
    alpha = 0.3
    scores = cost_aware_ei + alpha * exploration
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add tiny deterministic noise to break ties and encourage diversity
    # Use a hash-like deterministic perturbation based on index
    idx = np.arange(len(scores))
    scores += 1e-10 * np.sin(idx * 0.1)
    
    return scores
# EVOLVE-BLOCK-END
