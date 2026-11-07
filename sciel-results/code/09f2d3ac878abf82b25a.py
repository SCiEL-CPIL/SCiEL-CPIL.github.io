"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 295.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,)
        sigma: array of shape (2000,)
        f_best: scalar, lowest observed value
        cost: array of shape (2000,)
        
    Returns:
        array of shape (2000,) with scores (higher is better)
    """
    # Deadline guard: if we are close to the limit, use a fast heuristic
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Fast path: prioritize low mu and low cost
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # Thompson Sampling: sample from the Gaussian posterior
    # Use a fixed seed per call for reproducibility within a run,
    # but vary the sample to encourage exploration
    # We use a simple deterministic "sample" that approximates TS
    # by adding a scaled random component. However, since we cannot
    # use randomness that changes between calls in a way that's
    # hard to debug, we use a fixed perturbation based on a hash-like
    # approach or just a single sample from N(mu, sigma).
    
    # For efficiency and to avoid statefulness issues, we use a
    # deterministic approximation: sample = mu + sigma * z_sample
    # where z_sample is a fixed quantile that balances exploration.
    # A common choice is the 0.5 quantile (median), which is just mu.
    # To get exploration, we can use a higher quantile.
    
    # Alternatively, we can combine EI with a Thompson Sampling-like term.
    # Let's try: score = (ei + ts_bonus) / cost
    # where ts_bonus encourages exploration.
    
    # A better approach: use the mean of the posterior as the "sample"
    # but add a term that scales with uncertainty.
    
    # Let's stick to the core idea: EI is already a good balance.
    # The inspiration suggests Thompson Sampling. Let's implement a
    # simple version: sample = mu + sigma * randn. But we can't use
    # randn directly as it changes every call, which might be too noisy.
    
    # Instead, let's use a fixed "sample" from the posterior.
    # We can use the 0.9 quantile for more exploration.
    # z_0.9 = norm.ppf(0.9) ≈ 1.2816
    # ts_sample = mu + sigma * 1.2816
    # ts_score = -ts_sample  (since lower is better)
    
    # But this might be too exploratory. Let's blend EI and TS.
    
    # Actually, the simplest and most robust approach is to use EI
    # as the primary signal and add a small exploration bonus.
    
    # Let's try: score = (ei + 0.1 * sigma_safe) / cost_safe
    # This adds a small bonus for uncertainty.
    
    exploration_bonus = 0.1 * sigma_safe
    score = (ei + exploration_bonus) / cost_safe

    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
