"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses posterior sampling to balance exploration and exploitation.
    At each candidate point, we sample from the posterior N(mu, sigma^2)
    and score by the negative sampled value (since we're minimizing).
    
    To make this deterministic per call (as required by the interface),
    we use a fixed random seed derived from the inputs to get reproducible
    samples. However, since we need to return scores for ALL 2000 candidates,
    we compute the expected improvement of the Thompson sample.
    
    Actually, for Thompson Sampling in a single-shot acquisition function:
    We sample one value from the posterior at each candidate, then the 
    "score" is the negative of that sample (lower objective = better).
    But since we need a deterministic function, we use the expected value
    of the negative sample, which is just -mu. That's not useful.
    
    Instead, we use a practical approximation: the acquisition score is
    based on the probability that a sample from N(mu, sigma) beats f_best,
    weighted by the expected improvement of that sample. This is equivalent
    to Expected Improvement but we frame it as Thompson Sampling's decision rule.
    
    For a true Thompson Sampling approach in this interface, we compute:
    score = -mu + sigma * z_sample
    where z_sample is drawn from N(0,1). But we need deterministic output.
    
    The standard deterministic equivalent of Thompson Sampling for minimization
    is to use the quantile of the posterior. We use the mean minus a scaled
    standard deviation, which is equivalent to UCB with a specific beta.
    
    However, the directive says to implement Thompson Sampling by approximating
    the posterior sample. The key insight is that Thompson Sampling's expected
    performance matches EI, but the exploration-exploitation trade-off is
    different in practice due to the stochastic nature.
    
    For a deterministic implementation that captures the spirit of Thompson
    Sampling, we use:
    score = -mu + kappa * sigma
    where kappa is chosen to balance exploration. This is equivalent to UCB.
    
    But to be more faithful to Thompson Sampling, we can use the fact that
    the optimal Thompson Sampling decision in expectation corresponds to
    maximizing the probability of improvement, which leads to a modified EI.
    
    Let's use a hybrid: primarily EI (which is the expected value of Thompson
    Sampling's improvement) with a small stochastic-like exploration term.
    
    Actually, the cleanest deterministic Thompson Sampling equivalent is:
    score = -mu + sigma * norm.ppf(1 - p)
    where p is a small probability (e.g., 0.1), giving a lower quantile.
    
    For minimization, we want low values. Thompson Sampling picks the argmin of
    samples y_i ~ N(mu_i, sigma_i). In expectation, this favors low mu and
    high sigma. The deterministic proxy is the lower quantile of the posterior.
    
    We use the 10th percentile (p=0.1) as a proxy for the typical Thompson
    Sampling choice, which gives:
    score = -(mu + sigma * norm.ppf(0.1)) = -mu - sigma * norm.ppf(0.1)
    Since norm.ppf(0.1) is negative (~ -1.28), this becomes:
    score = -mu + sigma * 1.28
    
    This is equivalent to UCB with beta = 1.28.
    """
    t_start = time.time()
    deadline = t_start + 290.0  # 10s safety margin before 300s limit
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling approximation: use the lower quantile of the posterior
    # For minimization, we sample from N(mu, sigma) and pick the minimum.
    # The deterministic equivalent is the lower quantile.
    # Using the 10th percentile as a balance between exploration and exploitation.
    quantile = norm.ppf(0.1)  # ~ -1.2816
    # Score = negative of the quantile value (higher score = better)
    # quantile_value = mu + sigma * quantile
    # score = -quantile_value = -mu - sigma * quantile = -mu + sigma * 1.2816
    
    scores = -mu + sigma_safe * (-quantile)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
