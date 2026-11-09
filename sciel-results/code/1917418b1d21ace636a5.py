"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores. Higher scores indicate better candidates.
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Strategy:
    # Instead of computing a deterministic bound (like UCB or EI), we sample
    # a single function value from the posterior distribution for each candidate.
    # For f(x) ~ N(mu(x), sigma(x)^2), a sample is f_sample = mu(x) + sigma(x) * z,
    # where z ~ N(0, 1).
    
    # Since we only need to return scores for the evaluator to pick the *best* one,
    # and the evaluator will pick the one with the highest score, we can simply
    # return the sampled values (adjusted for cost) as the scores.
    # However, the prompt says "Return 2000 finite scores (higher = evaluate next)".
    # The evaluator picks the argmax of these scores.
    
    # If we just return -f_sample / cost, the evaluator will pick the candidate
    # with the lowest sampled function value relative to its cost.
    # This is exactly Thompson Sampling: sample from the posterior, pick the best.
    
    # Generate a single standard normal random number z.
    # Using a fixed seed per call is not possible as we don't have state, 
    # but we can use the global random state or just np.random.
    # Note: The prompt implies this function is called once per iteration to score 2000 points.
    # We want to sample *one* function surface (or effectively independent samples? 
    # Usually Thompson Sampling samples a function f(x) for all x, then picks x*.
    # If we sample independent z for each x, it's "independent Thompson Sampling" which is less standard 
    # but often used in BO when you can't sample the whole function. 
    # However, since we have mu and sigma for all 2000 points, and we want to pick ONE point,
    # sampling a single z and applying it to all is not quite right because the correlation structure is lost.
    # But standard TS for BO picks x* = argmin_x f_k(x) where f_k is a sample from the GP.
    # Since we can't sample the whole GP here (we only have pointwise mu/sigma), 
    # we approximate by sampling f(x_i) ~ N(mu_i, sigma_i^2) independently for each i.
    # This is a common approximation when full GP sampling is too expensive.
    
    # Let's generate independent samples for each candidate.
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    f_samples = mu + sigma * z
    
    # We want to minimize f(x). 
    # The "score" should be high for candidates that are likely to be good (low f) and cheap.
    # A common cost-aware TS score is: Score = -f_sample / cost
    # Or Score = (f_best - f_sample) / cost. 
    # Using (f_best - f_sample) / cost centers the improvement relative to current best.
    # If f_sample < f_best, score is positive. If f_sample > f_best, score is negative.
    # This encourages exploration if f_sample is much lower than f_best, even if mu is high (due to sigma*z being negative).
    
    score = (f_best - f_samples) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
