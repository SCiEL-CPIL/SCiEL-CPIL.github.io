# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t_start = time.time()
    deadline = t_start + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-10)

    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -12, 12)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(improvement > 0, ei, 0.0)

    # UCB (Upper Confidence Bound) - for minimization, we want low mu + high sigma
    # UCB score = -mu + kappa * sigma  (higher is better)
    # Scale kappa adaptively: larger when f_best is far from 0, smaller when close
    # f_best starts around ~100+ for Beale, minimum is 0
    # Normalize: progress = 1 - f_best / f_initial_estimate
    # We don't know f_initial, but we can use a heuristic:
    # If f_best > 10, we're early; if f_best < 1, we're late
    if f_best > 50:
        kappa = 3.0   # heavy exploration early
    elif f_best > 10:
        kappa = 2.0
    elif f_best > 3:
        kappa = 1.5
    elif f_best > 1:
        kappa = 1.0
    else:
        kappa = 0.5   # exploit when close to optimum

    ucb = -mu + kappa * sigma_safe

    # Thompson Sampling-like: sample from N(mu, sigma) and score by -sample
    # We can't actually sample per-candidate in a deterministic way that helps,
    # but we can approximate the expected value of the minimum of K samples:
    # E[min of K draws from N(mu_i, sigma_i)] ≈ mu_i - sigma_i * c(K)
    # where c(K) is the expected minimum of K standard normals
    # For K=1: c=0, K=2: c≈0.56, K=3: c≈0.85, K=5: c≈1.16, K=10: c≈1.54
    # This is like a "soft" UCB that naturally balances
    K = 3
    c_K = 0.85  # expected min of 3 standard normals
    ts_score = -(mu - c_K * sigma_safe)  # = -mu + c_K*sigma

    # Normalize each component to [0, 1] range for blending
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / rng

    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    ts_norm = normalize(ts_score)

    # Adaptive weights: blend EI, UCB, and TS
    # Early: more exploration (UCB, TS); Late: more exploitation (EI)
    if f_best > 50:
        w_ei, w_ucb, w_ts = 0.2, 0.5, 0.3
    elif f_best > 10:
        w_ei, w_ucb, w_ts = 0.35, 0.4, 0.25
    elif f_best > 3:
        w_ei, w_ucb, w_ts = 0.45, 0.3, 0.25
    elif f_best > 1:
        w_ei, w_ucb, w_ts = 0.55, 0.2, 0.25
    else:
        w_ei, w_ucb, w_ts = 0.7, 0.1, 0.2

    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_ts * ts_norm

    # Add a small raw EI component so that the absolute scale matters
    # (normalized scores can lose the magnitude of improvement)
    scores += 0.1 * ei

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
