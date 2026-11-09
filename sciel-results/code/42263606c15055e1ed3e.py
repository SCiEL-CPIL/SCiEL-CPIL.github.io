"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
DEADLINE_MARGIN = 10  # seconds

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function:
    1. If f_best is small (< 1.0), we are close to the minimum (0).
       Use pure exploitation: maximize -mu (minimize mu).
       This avoids the conservative bias of EI/LCB near the optimum.
    2. Otherwise, use Expected Improvement (EI) for a balanced exploration/exploitation trade-off.
       EI for minimization: E[max(f_best - f(x), 0)]
       We return -EI so that higher scores correspond to higher improvement.
    """
    # Check deadline
    if time.time() - _start_time > 330 - DEADLINE_MARGIN:
        # Fallback to pure exploitation if running out of time
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    if f_best < 1.0:
        # Exploitation phase: just follow the GP mean
        # We want to minimize mu, so we return -mu
        score = -mu
    else:
        # Exploration/Balanced phase: Expected Improvement
        # z = (f_best - mu) / sigma
        z = (f_best - mu) / sigma
        
        # Normal CDF and PDF
        # Use scipy.stats.norm if available, but numpy has no built-in cdf/pdf for normal.
        # We can use an approximation or import scipy. The prompt says SciPy is available.
        from scipy.stats import norm
        
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # This is the expected amount by which f(x) improves upon f_best.
        # Since we want to maximize this improvement, and the evaluator picks the highest score,
        # we can return EI directly.
        # However, EI is 0 where f(x) is likely worse than f_best.
        # Let's verify the sign.
        # If mu < f_best, z > 0, EI > 0.
        # If mu > f_best, z < 0, EI is small but positive (due to uncertainty).
        
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        
        # To be consistent with "higher score means evaluate next", and EI is positive for improvement,
        # returning EI directly is correct.
        score = ei
        
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
