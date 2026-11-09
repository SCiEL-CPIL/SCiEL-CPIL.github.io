"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin from 330s limit

def _check_deadline():
    """Returns True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (exploit) with
    a scaled uncertainty bonus (explore) and a surprise/information-gain term.
    
    For minimization:
    - EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    - Exploration bonus: kappa * sigma (scaled by current best)
    - Surprise term: encourages evaluating points with high predictive uncertainty
      relative to the scale of the problem
    
    Score = w_ei * EI + w_explore * (f_best + kappa * sigma)
    Higher score means evaluate next.
    """
    if _check_deadline():
        # Fallback: pure exploit
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Ensure sigma is non-negative and has a small floor to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # CDF and PDF of standard normal
    Phi = norm.cdf(z)
    phi = norm.pdf(z)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi

    # Exploration term: scale with the magnitude of f_best to adapt
    # When f_best is close to 0 (optimal), reduce exploration
    # When f_best is large, allow more exploration
    scale = np.abs(f_best) + 1e-6
    kappa = 1.5  # Exploration coefficient
    
    explore_bonus = kappa * sigma_safe * np.sqrt(scale)

    # Surprise / information gain term:
    # Points with very high sigma relative to the range of mu are "surprising"
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-10 else 1.0
    surprise = sigma_safe / (mu_range + 1e-6)
    surprise_bonus = 0.5 * surprise

    # Combine: weight EI heavily, add exploration and surprise
    # Normalize EI to be comparable in scale
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_normalized = ei / ei_max

    scores = 0.7 * ei_normalized + 0.2 * (explore_bonus / (np.max(explore_bonus) + 1e-10)) + 0.1 * (surprise_bonus / (np.max(surprise_bonus) + 1e-10))

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
