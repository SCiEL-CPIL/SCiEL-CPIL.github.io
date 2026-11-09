"""Rewritten program for Bayesian optimization of Styblinski-Tang 1D using Knowledge Gradient."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 15.0  # seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Knowledge Gradient (KG) with Monte Carlo approximation.
    
    The KG criterion approximates the expected improvement in the global minimum
    by sampling from the posterior predictive distribution at top-k candidates.
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple EI if too close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    
    # Step 1: Rank candidates by a preliminary score (EI or UCB) to select top-k
    # Use EI for the preliminary ranking as it's a good balance
    z_prelim = (f_best - mu) / (sigma + 1e-9)
    ei_prelim = (f_best - mu) * norm.cdf(z_prelim) + sigma * norm.pdf(z_prelim)
    
    # Select top-k candidates for KG computation
    # k should be small enough for computational efficiency but large enough for diversity
    k = min(50, n)
    top_k_indices = np.argsort(ei_prelim)[-k:]
    
    # Initialize scores array
    scores = np.zeros(n)
    
    # For candidates not in top-k, use the preliminary EI score scaled
    scores[~np.isin(np.arange(n), top_k_indices)] = ei_prelim[~np.isin(np.arange(n), top_k_indices)]
    
    # Step 2: For top-k candidates, compute KG via Monte Carlo
    # KG(x) = E[min(0, min_{y in Y} f(y) - f(x))] - min(0, f_best - f(x))
    # Approximate with Monte Carlo:
    #   Sample f(x) ~ N(mu(x), sigma(x))
    #   Sample f(y) ~ N(mu(y), sigma(y)) for y in top-k
    #   Compute min over y, then take expectation of min(0, min_y - f(x))
    
    n_samples = 100  # Number of Monte Carlo samples
    
    top_k_mu = mu[top_k_indices]
    top_k_sigma = sigma[top_k_indices]
    
    # Generate samples: shape (n_samples, k)
    rng = np.random.default_rng(42)
    samples_x = rng.normal(top_k_mu, top_k_sigma + 1e-9, size=(n_samples, k))
    
    # For each sample, compute min over k candidates
    min_samples = np.min(samples_x, axis=1)  # shape (n_samples,)
    
    # KG estimate for each top-k candidate i:
    # E[min(0, min_y - f_i)] where f_i is the i-th candidate's value
    # We need to compute this for each i in top_k
    
    kg_values = np.zeros(k)
    
    for i in range(k):
        # f_i samples: shape (n_samples,)
        f_i_samples = samples_x[:, i]
        
        # min(0, min_y - f_i) for each sample
        # min_y is min_samples, f_i is f_i_samples
        improvement = np.minimum(0, min_samples - f_i_samples)
        
        # The baseline term: min(0, f_best - f_i)
        # This is deterministic given f_i, but we average over samples
        baseline = np.minimum(0, f_best - f_i_samples)
        
        # KG = E[improvement] - E[baseline]
        kg_values[i] = np.mean(improvement) - np.mean(baseline)
    
    # Assign KG values to top-k candidates
    scores[top_k_indices] = kg_values
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
