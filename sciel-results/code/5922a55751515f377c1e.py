# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_t_start = None

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()
    
    # Safety: if we're running out of time, return a simple but valid score
    elapsed = time.time() - _t_start
    if elapsed > 310.0:
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        pi = norm.cdf(z)
        score = pi / (cost + eps)
        return np.where(np.isfinite(score), score, 0.0)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Structural change: use a cost-aware variant of Expected Improvement
    # that explicitly balances the magnitude of expected improvement against
    # cost, rather than just probability of improvement.
    #
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    #
    # This captures both the probability AND the magnitude of improvement.
    # For Rastrigin's multi-modal landscape, EI's sensitivity to the size
    # of potential gains helps it identify promising regions that PI might
    # miss when many regions have similar P(improvement) but very different
    # expected gains.
    #
    # Cost-awareness: divide by a power of cost to create a trade-off.
    # Using cost^0.5 (square root) provides a gentler penalty than cost^1,
    # allowing the algorithm to still explore moderately expensive regions
    # if the expected improvement is substantial.
    
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost^0.5
    # This balances expected gain against evaluation cost.
    cost_safe = np.maximum(cost, eps)
    score = ei / np.sqrt(cost_safe)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the multi-modal Rastrigin landscape
    # by giving some credit to regions where sigma is large, even if
    # the current mean is not promising.
    # The bonus is scaled relative to the typical EI magnitude.
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > eps:
        exploration_bonus = 0.1 * sigma_safe / sigma_mean
        score = score + exploration_bonus / np.sqrt(cost_safe)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
