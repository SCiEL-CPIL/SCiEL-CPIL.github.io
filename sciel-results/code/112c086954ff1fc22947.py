"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for the deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])

    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we are running too long, return greedy scores (-mu)
    # to ensure we finish within the time limit.
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Greedy: pick the point with the lowest mean prediction
        # Score = -mu (higher is better)
        return -mu

    # Increment call count to track optimization progress
    _call_count += 1
    
    # Expected Improvement (EI) for minimization
    delta = f_best - mu
    z = delta / sigma
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = delta * phi_z_cdf + sigma * phi_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - beta * sigma(x)
    # Higher LCB score means more promising for minimization
    # We use an adaptive beta that decreases as optimization progresses
    # Early on: high beta for exploration, Later: low beta for exploitation
    
    # Total evaluations = 30 (5 initial + 25 more)
    # _call_count goes from 1 to 25 (after initial 5)
    # Progress: 0.0 at start, 1.0 at end
    progress = min(_call_count / 25.0, 1.0)
    
    # Beta starts high (exploration) and decreases to low (exploitation)
    # Use exponential decay for smooth transition
    beta = 1.5 * np.exp(-2.0 * progress) + 0.1
    
    # LCB score for minimization: we want to minimize mu - beta*sigma
    # So higher score = more promising
    lcb_score = -mu + beta * sigma
    
    # Combine EI and LCB with adaptive weighting
    # Early on: weight more toward LCB for exploration
    # Later on: weight more toward EI for exploitation
    # Weight for EI increases with progress
    ei_weight = 0.3 + 0.7 * progress
    lcb_weight = 1.0 - ei_weight
    
    # Normalize both to have similar scale before combining
    # EI is already non-negative, LCB can be negative
    # Shift LCB to be non-negative for better combination
    lcb_min = np.min(lcb_score)
    lcb_shifted = lcb_score - lcb_min + 1.0  # Ensure positive
    
    # Scale EI to be comparable to LCB
    ei_scale = np.max(ei) if np.max(ei) > 1e-8 else 1.0
    ei_scaled = ei / ei_scale
    
    # Combine the two acquisition functions
    combined = ei_weight * ei_scaled + lcb_weight * lcb_shifted
    
    # Ensure finite values
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
