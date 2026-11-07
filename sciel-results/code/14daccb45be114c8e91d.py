"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Thompson Sampling acquisition with cost normalization
# We sample a single theta per acquisition call to balance exploration/exploitation
# Score is -theta / cost, so lower cost and lower theta (better objective) are preferred

# Cache for the random state to ensure reproducibility within a single acquisition call
# but we don't want to reset it between calls as that would make the search deterministic
# in a way that might not be ideal. However, for Thompson Sampling, we just need
# a single sample per call.

# To avoid issues with global state, we'll just use np.random directly.
# But we need to be careful: the problem says "No coordinates are passed", so we
# cannot use the input to seed anything specific. We just draw from the distribution.

# The key insight: we sample theta ~ N(mu, sigma) for ALL points simultaneously?
# No, that would make all points have the same uncertainty realization, which is wrong.
# Actually, the standard Thompson Sampling approach for a batch of candidates is:
# For each candidate i, sample theta_i ~ N(mu_i, sigma_i), then pick the one with best score.
# But here we are returning scores for 2000 candidates, and the evaluator picks the best.
# So we should sample theta_i for each candidate i independently?
# Or do we sample a single function sample?
#
# In Bayesian Optimization with Thompson Sampling, you typically sample a single function
# from the posterior and then optimize that sample. But here we are given mu and sigma
# for 2000 fixed points. The standard approach in this context (where we can't optimize
# the sample, only score the given points) is to sample theta_i ~ N(mu_i, sigma_i) for
# each point i, and score based on that.
#
# However, if we sample independently for each point, the exploration is "local" to each point.
# A more principled approach would be to sample a single value from the posterior process,
# but since we don't have the full GP kernel matrix, we can't easily do that.
#
# The directive says: "sampling a single scalar value θ from a Gaussian distribution with
# mean μ and standard deviation σ for all 2000 candidates". This is a bit ambiguous.
# Does it mean θ_i ~ N(μ_i, σ_i) for each i? Or θ ~ N(μ, σ) where μ and σ are vectors?
# The latter doesn't make sense for a single scalar.
#
# Re-reading: "sampling a single scalar value θ from a Gaussian distribution with mean μ
# and standard deviation σ for all 2000 candidates". This likely means: for each candidate,
# we have a Gaussian N(μ_i, σ_i), and we sample one value from it. So θ_i ~ N(μ_i, σ_i).
#
# Let's implement that.

def acquisition(mu, sigma, f_best, cost):
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample theta_i ~ N(mu_i, sigma_i)
    # Use a single random draw for all points to maintain correlation structure
    # (though in this simplified version, we're just sampling independently)
    theta = mu + sigma * np.random.standard_normal(len(mu))
    
    # Score: higher is better. We want low theta (good objective) and low cost.
    # Score = -theta / cost
    scores = -theta / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
