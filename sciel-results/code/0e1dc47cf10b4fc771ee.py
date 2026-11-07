"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with
    thinning-based batch correction.
    
    Structural change from the parent:
    1. Replace deterministic EI with stochastic Thompson Sampling
       (sample from the posterior at each candidate)
    2. Apply thinning penalty to model batch evaluation effects
    3. Use cost^0.8 normalization for budget awareness
    """
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample a function value from the posterior at each candidate
    # f(x) ~ N(mu, sigma^2)
    # The improvement is f_best - f(x), so we want high samples of (f_best - f(x))
    # i.e., low samples of f(x)
    
    # Sample from the posterior
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    
    # Compute the improvement for each sampled value
    # improvement = f_best - sample (positive means the sample is better than f_best)
    improvement = f_best - samples
    
    # Clip to non-negative (only count actual improvements)
    improvement = np.maximum(improvement, 0.0)
    
    # Thinning-based approximation: subtract a fraction of the top-k improvements
    # to model the probability that this point will remain the best after
    # a batch of evaluations. This shifts from immediate local improvement
    # to a more global view.
    k = 5
    if improvement.size > k:
        # Get the top-k improvement values
        top_k = np.partition(improvement, -k)[-k:]
        # Subtract a fraction of the mean top-k improvement to penalize points
        # that are likely to be surpassed by other high-improvement points
        thinning_penalty = 0.1 * np.mean(top_k)
        improvement_thinned = improvement - thinning_penalty
        improvement_thinned = np.maximum(improvement_thinned, 0.0)
    else:
        improvement_thinned = improvement
    
    # Apply cost-aware penalty with softened exponent
    # Using cost^0.8 allows selecting points with higher potential improvement
    # even if they are slightly more expensive, while still respecting the budget
    score = improvement_thinned / (c ** 0.8)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
