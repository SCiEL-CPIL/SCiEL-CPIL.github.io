"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for scheduling
_acq_call_count = 0
_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin from 330s
TOTAL_ACQ_CALLS = 25  # 30 total evals - 5 initial

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition using a global schedule to blend Expected Improvement (EI)
    and Lower Confidence Bound (LCB).
    
    Early iterations favor LCB for exploration.
    Later iterations favor EI for exploitation.
    Both components are normalized to zero mean and unit variance to ensure
    comparable scales regardless of the current f_best or sigma magnitudes.
    """
    global _acq_call_count, _start_time
    
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Return pure exploitation (minimize mu) as a safe fallback
        return -np.asarray(mu, dtype=np.float64).copy()

    _acq_call_count += 1
    k = _acq_call_count
    if k > TOTAL_ACQ_CALLS:
        k = TOTAL_ACQ_CALLS

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # 2. Compute Lower Confidence Bound (LCB) score
    # For minimization, we want to minimize (mu - beta * sigma).
    # To convert to a "higher is better" score, we use -(mu - beta * sigma) = -mu + beta * sigma.
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both scores to zero mean and unit variance
    # This prevents one component from dominating due to scale differences
    # that change as f_best decreases and sigma shrinks.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-8:
        ei_std = 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-8:
        lcb_std = 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # 4. Determine weights based on iteration count
    # Exponential decay of exploration (LCB) weight
    # w_lcb starts high and decays to 0
    # w_ei starts low and increases to 1
    decay_rate = 5.0
    w_lcb = np.exp(-k / decay_rate)
    w_ei = 1.0 - w_lcb
    
    # Blend
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
