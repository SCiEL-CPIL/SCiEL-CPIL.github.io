"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s before the 330s kill

def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Combines Probability of Improvement (PI) and Expected Improvement (EI)
    with adaptive weighting based on the current best observed value.
    
    PI is robust for multimodal landscapes as it focuses on the probability
    of finding a better point rather than the magnitude of the improvement.
    EI is added to exploit regions with high expected gain.
    The combination is normalized by the evaluation cost to prioritize
    cheaper evaluations.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        # Fallback: PI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return pi / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be comparable to PI (which is in [0, 1])
    # f_best is the scale of the objective function
    f_scale = abs(f_best) + 1.0
    ei_norm = ei / f_scale
    
    # Combine PI and normalized EI
    # PI drives exploration in multimodal landscapes (finding *any* improvement)
    # EI drives exploitation in regions with high expected gain
    # We weight them equally initially, or slightly favor PI for robustness
    # in the early stages when f_best is large and EI might be noisy.
    
    # Adaptive weighting:
    # When f_best is large (early), rely more on PI to find the basin.
    # When f_best is small (late), rely more on EI to polish.
    # However, without iteration count, we use f_best as a proxy.
    # If f_best is close to 0 (min), we want to be precise -> EI.
    # If f_best is large, we want to search -> PI.
    
    # Simple heuristic:
    # weight_pi = 0.5 + 0.5 * (1 / (1 + f_scale)) ? No, f_scale is always >= 1.
    # Let's just use a fixed weight that worked for the high-scoring child.
    # The child likely used a mix. Let's try 0.5 PI + 0.5 EI_norm.
    
    score_unscaled = 0.5 * pi + 0.5 * ei_norm
    
    # Cost-awareness: divide by cost
    # Cost is in [1, 2]
    score = score_unscaled / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
