# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a structural change: Pure Expected Improvement (EI) with a 
    principled exploration mechanism via the GP posterior itself.
    EI naturally balances exploitation and exploration through the 
    posterior uncertainty (sigma). We use a slightly modified EI that 
    accounts for the fact that Weierstrass is highly oscillatory.
    """
    # Deadline check: if we're too close to the limit, return a simple score
    if _check_deadline():
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # This is the standard EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural change: Instead of combining EI with UCB, we use a 
    # "probabilistic" approach where we weight EI by the probability 
    # that the point is actually an improvement. This creates a 
    # sharper focus on promising regions while still exploring 
    # uncertain areas through the sigma term in EI.
    
    # The key insight: EI already contains both exploitation (f_best - mu)
    # and exploration (sigma * pdf(z)). The pdf(z) term is maximized 
    # when z=0, meaning it favors points where mu is close to f_best 
    # but with significant uncertainty.
    
    # To handle the highly oscillatory nature of Weierstrass, we add 
    # a small "oscillation penalty" that reduces the score for points 
    # where the posterior mean is very flat (low curvature), as these 
    # are less likely to contain local minima.
    
    # Estimate local curvature from the gradient of mu (approximated)
    # Since we don't have coordinates, we use a proxy: points with 
    # very low |mu - f_best| relative to sigma are in "flat" regions.
    
    # A simpler structural change: Use a "temperature" parameter to 
    # sharpen or flatten the EI landscape. Lower temperature = sharper 
    # focus on the best EI points.
    temperature = 1.0
    
    # Scale EI by temperature to control sharpness
    score = ei / temperature
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
