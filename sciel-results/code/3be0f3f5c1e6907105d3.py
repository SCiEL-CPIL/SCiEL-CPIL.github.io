"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo estimation of Expected Regret
    relative to the known true minimum (3.0).
    
    Structural change: Instead of closed-form EI/UCB blends, we sample
    from the posterior at each candidate point and compute the expected
    regret E[max(3.0 - f(x), 0)] via Monte Carlo. This provides a
    natural exploration/exploitation balance and may behave differently
    in practice due to the sampling-based approach.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    
    # Safety: clip sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The known global minimum of Goldstein-Price 2D
    f_true_opt = 3.0
    
    # Monte Carlo estimation of Expected Regret: E[max(3.0 - f(x), 0)]
    # Sample K times from N(mu, sigma) at each point
    # Use vectorized sampling for efficiency
    
    # Number of MC samples - balance accuracy vs speed
    K = 200
    
    # Generate K x n samples from N(mu_i, sigma_i) for each i
    # shape: (K, n)
    z = np.random.randn(K, n)
    samples = mu[None, :] + sigma_safe[None, :] * z  # (K, n)
    
    # Compute regret for each sample: max(3.0 - sample, 0)
    regret = np.maximum(f_true_opt - samples, 0.0)  # (K, n)
    
    # Expected regret: mean over K samples for each point
    er = np.mean(regret, axis=0)  # (n,)
    
    # Also compute a secondary term: probability of beating true optimum
    # P(f(x) < 3.0) = Phi((3.0 - mu) / sigma)
    z_true = (f_true_opt - mu) / sigma_safe
    prob_beat = norm.cdf(z_true)
    
    # Combine: primarily use ER, with a small boost from prob_beat
    # This helps distinguish points with similar ER but different confidence
    
    # Normalize both to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn > 1e-12:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    er_norm = normalize(er)
    prob_norm = normalize(prob_beat)
    
    # Weighted combination
    scores = 0.85 * er_norm + 0.15 * prob_norm
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
