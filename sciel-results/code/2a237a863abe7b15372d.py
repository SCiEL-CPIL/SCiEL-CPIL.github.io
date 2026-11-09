"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We use a deterministic pseudo-random seed for the Thompson sampling noise
# to ensure reproducibility within a single acquisition call, but the key is
# that we sample from the posterior. However, standard Thompson Sampling for
# acquisition functions usually involves sampling from the posterior of the
# function value at candidate points.
#
# The directive suggests: (f_best - mu) / (sigma + 1e-9) / cost
# This is the z-score of improvement, normalized by cost.
# Wait, standard Thompson Sampling for minimization:
# We want to maximize P(f(x) < f_best).
# Or we can sample y ~ N(mu, sigma^2) and score = (f_best - y) / cost.
# The expectation of (f_best - y) / cost is (f_best - mu) / cost.
# The variance introduces exploration.
#
# Let's stick to the specific suggestion in the prompt but refine it.
# "Thompson Sampling posterior sample of the inverse regret"
# Regret R(x) = f(x) - f_best (we want to minimize, so negative regret is good? No, usually regret is f(x) - f_min).
# Here, improvement I(x) = f_best - f(x).
# We want to maximize expected improvement per cost.
# Thompson Sampling: Sample f(x) ~ N(mu, sigma^2).
# Score = max(0, f_best - f_sample) / cost.
#
# However, computing a sample for 2000 points is fast.
# Let's try:
# 1. Sample noise: epsilon ~ N(0, 1)
# 2. f_sample = mu + sigma * epsilon
# 3. improvement = f_best - f_sample
# 4. score = improvement / cost
# 5. To avoid negative scores dominating (though we want to pick the best, negative improvement means it's worse than current best, but maybe we should still explore? Usually EI is clipped at 0. If we don't clip, we might pick points that are likely worse but have low cost? The cost normalization handles that. But if improvement is negative, the score is negative. We want the MAX score. So negative scores are bad. This effectively clips at 0 implicitly if there are any positive scores. If all are negative, it picks the "least bad" expensive point? No, it picks the one with highest (f_best - f_sample)/cost. If f_sample > f_best, improvement is negative.
#
# Actually, the standard "Cost-aware Thompson Sampling" often uses:
# score = (f_best - mu) / (sigma * cost) + something?
#
# Let's look at the prompt's specific formula: `(f_best - mu) / (sigma + 1e-9) / cost`
# This is just the z-score divided by cost. This is a deterministic value (no random sampling).
# This is effectively "Expected Improvement per cost" approximated?
# No, EI is (f_best - mu) * Phi(z) + sigma * phi(z).
# The z-score is (f_best - mu) / sigma.
# Dividing by cost gives z / cost.
# This favors high z (high confidence of improvement) and low cost.
# It ignores the magnitude of the improvement (only relative to sigma).
#
# Let's try a hybrid:
# Use a sampled value from the GP posterior to create a stochastic acquisition.
# This is a structural change from the deterministic EI/cost.
#
# Implementation:
# 1. Generate a random vector of standard normals of shape (2000,).
# 2. Compute f_sample = mu + sigma * randn.
# 3. Compute improvement = f_best - f_sample.
# 4. Compute score = improvement / cost.
# 5. Return score.
#
# To ensure we don't pick a point that is clearly worse (negative improvement) if there are better options, the max operation in the caller will handle it. If all improvements are negative, it picks the one with the least negative score (i.e., highest improvement relative to cost).
#
# One issue: Randomness in acquisition functions can be unstable.
# However, Thompson Sampling is a valid bandit strategy.
#
# Let's add a deadline guard as requested.
_start_time = time.time()
_TIMEOUT = 300.0 # Seconds
_MARGIN = 10.0   # Safety margin

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _TIMEOUT - _MARGIN:
        # Fallback to a simple greedy-like score or just return zeros to stop?
        # The contract says "return 2000 finite scores".
        # If we time out, we should return something valid quickly.
        # A safe fallback is the deterministic z-score / cost.
        z = (f_best - mu) / (sigma + 1e-9)
        return z / cost

    # Thompson Sampling Approach
    # Sample from the posterior
    # Note: We need to be careful with the random seed or just use np.random.
    # Using a fixed seed per call might make it deterministic per call, but different calls
    # (if the function was called multiple times in a loop, which it isn't here, it's called once per iteration)
    # Actually, the function is called once per BO iteration.
    # So we can just use np.random.
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples
    # We want to maximize (f_best - f(x)) / cost
    # f(x) ~ N(mu, sigma^2)
    # Let u ~ N(0, 1)
    # f_sample = mu + sigma * u
    # Score = (f_best - (mu + sigma * u)) / cost
    # Score = ((f_best - mu) / cost) - (sigma / cost) * u
    
    # This is linear in u.
    # To make it more robust, we can clip the improvement to be non-negative?
    # Standard EI clips at 0. TS usually doesn't clip the sample, but the reward function might.
    # If we don't clip, we might select a point with negative expected improvement but high uncertainty and low cost.
    # Given the tight budget (30 cost units), exploration should be cautious.
    # Let's use the clipped version: max(0, f_best - f_sample) / cost.
    
    u = np.random.randn(len(mu))
    f_sample = mu + sigma_safe * u
    improvement = np.maximum(0.0, f_best - f_sample)
    score = improvement / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
