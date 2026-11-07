# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DELINE

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Strategy:
    - Instead of calculating a deterministic score like EI or UCB, we sample
      a single value from the posterior predictive distribution for each candidate.
      The posterior is approximated as Normal(mu, sigma).
    - We select the candidate that minimizes this sampled value.
    - Since the acquisition function must return a score where higher is better,
      we return the negative of the sampled value.
    
    Why this is different:
    - The current program uses a deterministic blend of EI and LCB, normalized by cost.
    - This approach uses stochastic sampling. This naturally balances exploration and
      exploitation without needing to tune hyperparameters like beta in UCB or blending
      weights. High uncertainty (sigma) leads to a wider spread of samples, increasing
      the chance that a low value is sampled (exploration), while low uncertainty
      focuses on the mean (exploitation).
    - Cost is handled by the external loop's budget constraint, not by penalizing
      the acquisition score. This avoids the potential issue of suppressing
      exploration in high-cost regions if the cost penalty is too aggressive.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values and safe ranges
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical instability
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Check deadline
    if _check_deadline():
        # If we are running out of time, return a deterministic fallback
        # to ensure we don't crash. We return the negative of the mean,
        # which is a reasonable greedy fallback.
        return -mu
    
    # Generate random samples from N(mu, sigma)
    # np.random.normal(loc=mu, scale=sigma)
    # To ensure reproducibility isn't strictly required but we want randomness
    # for Thompson Sampling.
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # The acquisition score is the negative of the sampled value.
    # We want to maximize this score, which corresponds to minimizing the sampled objective.
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
# EVOLVE-BLOCK-END
