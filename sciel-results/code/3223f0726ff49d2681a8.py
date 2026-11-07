import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization
    of Goldstein-Price 2D.
    
    Uses pure EI to balance exploration and exploitation.
    Adaptive xi parameter encourages exploitation when close to the minimum.
    """
    # Ensure inputs are arrays of correct type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Wall-clock guard: never let this call approach the kill limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast)
        # If time is up, just pick the most likely minimum (lowest mu)
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Calculate the gap to the known global minimum
    gap = max(f_best - f_min, 0.0)
    
    # Adaptive xi for EI
    # When gap is large, we might want a bit more exploration (higher xi? No, higher xi makes EI more conservative/exploratory in a sense? 
    # Actually, standard EI uses xi=0. 
    # If we want to explore, we can use a small positive xi to demand improvement.
    # If we want to exploit, xi should be 0 or very small.
    # However, in minimization, EI(x) = E[max(f_best - f(x), 0)].
    # If f(x) is likely > f_best, EI is 0.
    # If we want to explore areas where f(x) might be much lower, EI captures that.
    # The "exploration" in EI comes from the uncertainty (sigma).
    # A larger sigma increases the probability that f(x) < f_best, thus increasing EI.
    # So EI naturally explores high uncertainty regions.
    # To tune exploitation vs exploration, we can scale the "improvement" we expect.
    # Let's use xi = 0 for pure EI, as it is the standard and often robust choice.
    # Alternatively, we can add a small penalty for not improving significantly if we are stuck.
    # But let's stick to standard EI first as a structural change from the hybrid.
    
    # Let's try a small adaptive xi to encourage finding the minimum faster if we are close.
    # If gap is small, we want to be very precise. xi=0 is good.
    # If gap is large, maybe we want to ensure we find *some* improvement?
    # Actually, standard EI (xi=0) is usually best. Let's stick with xi=0.
    
    xi = 0.0
    
    # Calculate EI
    # z = (f_best - mu - xi) / sigma
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu - xi) / sigma_safe
    
    # Use stable calculation if possible, but standard formula is usually fine for 2000 points
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    
    ei_score = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI scores are finite
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    
    return ei_score
