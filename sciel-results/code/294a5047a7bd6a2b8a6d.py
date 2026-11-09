import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 300.0

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a normalized blend of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) with adaptive weighting, plus Thompson
    Sampling exploration in early stages.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to exploitation
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Calculate progress (0 to 1)
    step = (_acq_call_count - 1) % 25
    progress = step / 25.0
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF using scipy.special
    cdf_z = 0.5 * sp.erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Standard EI formula
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound) - for minimization, lower is better
    # LCB = mu - beta * sigma, we want high scores, so use -LCB = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Normalize both components to [0, 1] range for proper blending
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb_score)
    
    # Adaptive weighting: more LCB early (exploration), more EI later (exploitation)
    # Use exponential decay for smooth transition
    lcb_weight = np.exp(-3.0 * progress)
    ei_weight = 1.0 - lcb_weight
    
    # Blend the normalized components
    blended = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # In early stages, add Thompson Sampling exploration
    # This introduces randomness to help escape local optima
    if progress < 0.3:
        # Draw random samples from standard normal
        ts_noise = np.random.standard_normal(len(mu))
        # Scale the Thompson sampling contribution
        ts_scale = 0.5 * (1.0 - progress / 0.3)  # Decays to 0 at progress=0.3
        blended = blended + ts_scale * ts_noise * (sigma / np.max(sigma_safe) + 1e-9)
    
    # Rescale back to original EI scale for consistent magnitude
    ei_range = np.max(ei) - np.min(ei)
    if ei_range > 1e-12:
        score = np.min(ei) + blended * ei_range
    else:
        score = ei
    
    # Ensure finite scores
    score[~np.isfinite(score)] = 0.0
    
    return score
