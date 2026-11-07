"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and potential fallback
_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds to reserve before the 330s kill

def _check_deadline():
    """Check if we are close to the time limit."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    elapsed = time.time() - _START_TIME
    # The evaluator kills at 330s. We want to be safe.
    return elapsed > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing Expected Improvement (which is greedy and can get stuck in local optima),
    we sample from the posterior distribution of the objective function values at the candidate points.
    We then return the sampled values as the scores. The evaluator will pick the candidate with the
    highest score, which corresponds to the candidate with the highest sampled function value.
    
    This approach naturally balances exploration (high sigma leads to large variance in samples, 
    potentially picking unexplored regions) and exploitation (low mu leads to high mean samples).
    """
    n = mu.shape[0]
    
    # If we are running out of time, fall back to a very fast heuristic (e.g., UCB or just -mu)
    # to ensure we return something valid quickly.
    if _check_deadline():
        # Simple Upper Confidence Bound (UCB) as a fast fallback
        # UCB = mu + kappa * sigma
        # For minimization, we want to maximize the "potential to be low".
        # Actually, for minimization, EI is standard. Let's use a fast EI approximation or just -mu.
        # -mu is exploitation only. UCB with positive kappa encourages exploration.
        # Let's use a simple EI calculation which is O(N) and very fast.
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Thompson Sampling:
    # Draw one sample from the posterior for each candidate point.
    # Posterior at x_i: N(mu_i, sigma_i^2)
    # We assume independence for the sampling step because we don't have the full covariance matrix 
    # available in the function signature (only mu and sigma are passed).
    # This is a standard approximation when the full kernel/covariance is not accessible.
    
    # Generate random samples
    # Using standard normal and scaling
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(n)
    
    # The acquisition score for Thompson Sampling is the sampled value itself.
    # Since we are minimizing, the evaluator will pick the point with the HIGHEST score.
    # Wait, the contract says: "Return 2000 finite scores, higher meaning evaluate next".
    # And "evaluates the candidate with the highest score".
    # The objective is minimization.
    # If we return the sampled value `s_i`, the evaluator picks `argmax(s_i)`.
    # This means it picks the point where the GP *predicted* the function value to be the *highest*.
    # That is the opposite of what we want! We want to find the *lowest* function value.
    
    # Correction:
    # The acquisition function should return a score such that higher score = better candidate to evaluate.
    # For minimization, a "good" candidate is one with a low function value.
    # In Thompson Sampling, we sample `f_i ~ N(mu_i, sigma_i^2)`.
    # We want to select the `i` that minimizes `f_i`.
    # So the "score" that the evaluator should maximize should be `-f_i` (negative of the sample).
    # If we return `-samples`, the evaluator picks `argmax(-samples)` which is `argmin(samples)`.
    # This selects the point where the sampled function value is lowest.
    
    return -samples
# EVOLVE-BLOCK-END
