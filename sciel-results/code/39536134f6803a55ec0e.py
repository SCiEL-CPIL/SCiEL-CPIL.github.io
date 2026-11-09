import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time per acquisition call and bail out if we're running long.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s

def _check_deadline():
    """Return True if we should stop computing and return a safe fallback."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Bayesian Search Theorem (BST) inspired
    score: cost-discounted Probability of Improvement with an adaptive
    information-gain (uncertainty reduction) term.
    
    The score combines:
      1. PI: probability that the point improves on f_best (robust, multimodal-friendly)
      2. Info-gain proxy: sigma * (1 - PI) which rewards points with high posterior
         uncertainty in regions that haven't been confidently ruled out
      3. Cost normalization: divide by cost to prefer cheaper evaluations
    
    This structural change from pure PI/cost to a BST-inspired information-gain
    blend should better balance exploration in the asymmetric cost landscape.
    """
    n = len(mu)
    
    # Safety: ensure finite outputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score for PI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Information gain proxy: expected reduction in uncertainty of the global min.
    # Points with high sigma in regions where improvement is still possible (1 - pi)
    # carry the most information. This is a proxy for the entropy reduction of
    # the posterior over the global minimum location.
    info_gain = sigma_safe * (1.0 - pi)
    
    # Blend: weight PI heavily (it's the primary improvement signal) but add
    # information gain to encourage exploration of uncertain regions.
    # The 0.3 weight on info_gain provides a structural exploration bonus
    # beyond what pure PI offers.
    score = (pi + 0.3 * info_gain / (np.abs(f_best) + 1.0)) / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
