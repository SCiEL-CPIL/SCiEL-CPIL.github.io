"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition for 1D sphere minimization.
    
    Strategy: Sample 100 latent function values from the GP posterior at each
    candidate point, and return the negative of the minimum sampled value per
    candidate. This directly optimizes for the probability that a point
    contains the global minimum among the samples.
    
    To make this computationally feasible with 2000 candidates, we use a 
    sparse approximation: we select a subset of candidates (those with low 
    posterior mean or high uncertainty) and perform full TS on that subset,
    then extrapolate to all candidates using a simpler score.
    
    Actually, for 2000 points in 1D, Cholesky is O(n^3) = 8e9 ops which is too 
    slow. Instead, we use a hybrid approach:
    1. Compute a fast acquisition score (EI-based) for all 2000 candidates.
    2. Select the top K candidates by this score.
    3. For those K candidates, perform proper multi-sample TS.
    4. Return the TS-based scores for the selected candidates and EI-based 
       for the rest.
    
    But even K=200 TS with Cholesky is 8e6 ops, which is fine. Let's do K=500.
    
    Actually, let's think differently. The key insight is that in 1D with 2000
    uniformly spaced candidates, we can use a simpler approach: compute the
    posterior variance at each point, and use a variant of UCB that combines
    exploitation and exploration in a principled way.
    
    Let me use a proven approach: Expected Hypervolume Improvement (EHVI) 
    approximated via sampling, but since we only care about the minimum, 
    we can use a simpler multi-sample approach on a subset.
    
    For 1D sphere with 2000 candidates, let's use:
    - Compute EI for all candidates (fast)
    - Select top 100 by EI
    - For those 100, do 50-sample Thompson Sampling
    - Return the TS scores for top 100, EI for others
    """
    n = len(mu)
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Step 1: Compute a baseline acquisition score (EI) for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Step 2: Select top K candidates by EI for more careful analysis
    K = min(200, n)
    top_idx = np.argpartition(ei, -K)[-K:]
    
    # Step 3: For the top K, perform multi-sample Thompson Sampling
    # We need to sample from the GP posterior at these K points
    # The posterior at test points given training data:
    # mu_test = mu (already computed by the caller)
    # We need the covariance structure to sample properly
    
    # Since we don't have access to the training data or kernel parameters,
    # we approximate: the posterior std at each point is sigma.
    # For Thompson Sampling in 1D, we can sample f(x) ~ N(mu(x), sigma(x)^2)
    # independently at each point (ignoring correlation between test points)
    # This is a common approximation when we don't have the full covariance.
    
    # Sample 100 values from the posterior at each of the top K candidates
    n_samples = 100
    rng = np.random.default_rng()
    
    # f_samples[i, j] = sample j of function value at candidate i
    f_samples = mu[top_idx][:, np.newaxis] + sigma[top_idx][:, np.newaxis] * rng.standard_normal((K, n_samples))
    
    # For each candidate, the "Thompson Sampling score" is the negative of 
    # the minimum value across samples (since lower is better)
    ts_scores = -np.min(f_samples, axis=1)
    
    # Step 4: Combine scores
    scores = ei.copy()
    scores[top_idx] = ts_scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
