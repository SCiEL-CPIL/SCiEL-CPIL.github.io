"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with dynamic weighting based on posterior uncertainty and current best value.
    
    Strategy:
    - EI handles exploitation well when f_best is good.
    - UCB handles exploration well when uncertainty is high.
    - Dynamic blending: when mean sigma is high relative to f_best, favor UCB; otherwise favor EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, np.inf)
    
    # 1. Expected Improvement (EI) for minimization
    # EI(x) = (fb - mu) * Phi((fb - mu)/sigma) + sigma * phi((fb - mu)/sigma)
    z = (fb - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (fb - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # Score to maximize: beta * sigma - mu
    # Adaptive beta: higher when f_best is large (early stage), lower when f_best is small (late stage)
    # f_best ranges from ~25 (x^2 at x=5) down to 0
    # Use a smooth transition: beta = 1.0 + 2.0 * (fb / (fb + 1.0))
    # When fb=0: beta=1.0 (exploit)
    # When fb=1: beta=2.0 (balanced)
    # When fb=10: beta=2.82 (explore)
    # When fb=25: beta=3.0 (explore heavily)
    beta = 1.0 + 2.0 * (fb / (fb + 1.0))
    
    ucb_scores = beta * sigma_safe - mu
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    # 3. Normalize both to comparable scales for blending
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_scores - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_scores)
    
    # 4. Dynamic weighting based on uncertainty level
    # Mean sigma indicates overall uncertainty
    # If mean_sigma is high, we are in early/exploration phase -> favor UCB
    # If mean_sigma is low, we are in late/exploitation phase -> favor EI
    
    mean_sigma = np.mean(sigma_safe)
    
    # Sigmoid-like transition: 
    # At mean_sigma=0.1: w_ucb ~ 0.12 (mostly EI)
    # At mean_sigma=1.0: w_ucb ~ 0.5 (balanced)
    # At mean_sigma=5.0: w_ucb ~ 0.99 (mostly UCB)
    x = mean_sigma / 1.0
    w_ucb = 1.0 / (1.0 + np.exp(-x))
    
    # Additional dampening: if f_best is very small, strongly favor exploitation
    if fb < 0.01:
        w_ucb *= 0.3
    elif fb < 0.1:
        w_ucb *= 0.5
    elif fb < 1.0:
        w_ucb *= 0.7
    
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    w_ei = 1.0 - w_ucb
    
    # Blend normalized scores
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Rescale to a reasonable magnitude
    # The original scores had different magnitudes; scale back by the max of original maxes
    max_original = max(np.max(ei), np.max(ucb_scores))
    if max_original > 0:
        scores = scores * max_original
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
