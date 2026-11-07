import numpy as np
import time
from scipy.stats import norm

# Track elapsed time to enforce deadline
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds
_MARGIN = 5.0      # safety margin before limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small
    improvement threshold (xi) to encourage slightly more exploration
    than vanilla EI, while still remaining in the EI family.
    
    For minimization:
    EI_xi(x) = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - xi - mu) / sigma
    
    A small positive xi (e.g., 0.01 * sigma_scale) biases the acquisition
    toward points that are likely to improve by at least xi, which can
    help escape shallow local optima in multimodal landscapes.
    
    We also add a tiny UCB-like exploration bonus scaled by sigma to
    encourage visiting uncertain regions when EI is nearly zero everywhere,
    but keep it small enough not to dominate the EI signal.
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: Pure exploitation (minimize mu)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite to avoid numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Use a small improvement threshold xi as a fraction of the typical scale
    # of sigma to adapt the exploration bias to the current uncertainty level
    sigma_scale = np.median(sigma)
    if not np.isfinite(sigma_scale) or sigma_scale < 1e-10:
        sigma_scale = 1e-8
    xi = 0.01 * sigma_scale  # small threshold to encourage slightly more exploration
    
    # Calculate z-score for EI with threshold
    z = (f_best - xi - mu) / sigma
    
    # Calculate EI with threshold
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = (f_best - xi - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # Add a small UCB-like exploration bonus to encourage diversity when EI is low
    # The bonus is proportional to sigma and scaled down to not dominate EI
    ucb_bonus = 0.05 * sigma
    ucb_bonus = np.where(np.isfinite(ucb_bonus), ucb_bonus, 0.0)
    
    scores = ei_scores + ucb_bonus
    
    # Ensure final scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
