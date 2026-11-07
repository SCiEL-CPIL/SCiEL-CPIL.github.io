# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Combines Expected Improvement (EI) for exploitation with an adaptive
    Probability of Improvement (PI) exploration bonus that decays as f_best improves.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement for minimization
    # PI = Phi((f_best - mu) / sigma)
    # This favors points with high likelihood of surpassing current best
    pi = norm.cdf(z)
    
    # Adaptive exploration weight based on PI
    # Weight is higher when f_best is large (early) and lower when f_best is small (late)
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    
    # The PI bonus scales with the probability of improvement
    # and an adaptive coefficient that decays as we get closer to the optimum
    # We use a coefficient that starts around 1.0 and decays to 0.1
    beta_pi = 0.1 + 0.9 * np.sqrt(ratio)
    
    # PI exploration term: beta_pi * pi * sigma
    # This adds a bonus proportional to the probability of improvement
    # and the uncertainty (sigma), scaled by the adaptive coefficient
    pi_bonus = beta_pi * pi * sigma
    
    # Combine EI and PI bonus
    # EI dominates exploitation, PI provides a different exploration bias
    score = ei + pi_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei
    
    return score
# EVOLVE-BLOCK-END
