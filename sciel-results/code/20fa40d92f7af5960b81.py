"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s limit

def _check_deadline():
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function computes the expected improvement over the current best value f_best,
    weighted by the cost of evaluation. It uses a deterministic EI formulation which is
    generally more stable and effective for BO than single-sample Thompson Sampling,
    especially when the number of candidates is large (2000).
    
    The cost is incorporated by dividing the EI by the cost, effectively prioritizing
    improvements that are cheap to obtain.
    
    To handle the "exploration" aspect which EI naturally dampens as sigma -> 0,
    we do not add explicit noise, but the EI formulation itself provides a balance.
    However, standard EI can be too greedy. We use the standard EI formula.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score = EI(x) / cost(x)
    """
    if _check_deadline():
        # Fallback: return zero scores or simple EI if time is critical
        # Returning zeros might cause random selection, which is better than crashing
        # But let's try to return a quick EI calculation without overhead if possible
        # Actually, if we are near deadline, we should return quickly.
        # A simple heuristic: prefer low mu and high sigma? No, just return 0s to stop?
        # The contract says "Return 2000 finite scores". 
        # If we return 0s, the evaluator might pick index 0 (or random if ties).
        # Let's return a quick EI approximation to be safe, but it might be slow.
        # Better: return a pre-computed or very fast score.
        # Given the deadline, let's just return -mu / cost as a fast proxy for "good value".
        # This is a valid finite score.
        cost_safe = np.maximum(cost, 1e-6)
        return -mu / cost_safe

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If f_best - mu is large positive, z is large positive -> Phi(z) ~ 1, phi(z) ~ 0 -> EI ~ f_best - mu
    # If f_best - mu is negative, z is negative -> Phi(z) ~ 0, phi(z) > 0 -> EI ~ sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails, though scipy norm is robust
    # norm.cdf and norm.pdf handle large values well, but clipping prevents inf/nan in edge cases
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: higher EI per unit cost is better
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
