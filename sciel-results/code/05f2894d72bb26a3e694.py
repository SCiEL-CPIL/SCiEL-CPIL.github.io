"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = time.time() + 325.0


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return time.time() >= _DEADLINE


def _kg_score(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition: expected reduction in the
    best observed value after evaluating at x.
    
    KG(x) = E[max(f_best - f(x), 0)] - E[min(f(x), f_best)]
         = E[f(x) - min(f(x), f_best)]
         = E[max(f(x) - f_best, 0)] ... wait, let me be precise.
    
    Actually, KG(x) = E[f_best] - E[min(f_best, f(x))]
                   = E[max(f_best - f(x), 0)]
    
    For f(x) ~ N(mu, sigma^2):
    E[max(f_best - f(x), 0)] = (f_best - mu) * Phi((f_best - mu)/sigma)
                              + sigma * phi((f_best - mu)/sigma)
    
    This is the expected improvement of the global minimum estimate,
    which is more principled than EI for multi-modal landscapes because
    it accounts for the probability that x becomes the new global minimum.
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    
    # E[max(f_best - f(x), 0)] for f(x) ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    kg = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite and non-negative
    kg = np.where(np.isfinite(kg), kg, 0.0)
    kg = np.maximum(kg, 0.0)
    
    return kg


def _ucb_score(mu, sigma, f_best, beta=2.0):
    """
    Upper Confidence Bound (UCB) for minimization:
    UCB(x) = -mu(x) + beta * sigma(x)
    
    Higher score means more promising (lower predicted value with uncertainty bonus).
    """
    return -mu + beta * sigma


def _blend_scores(mu, sigma, f_best):
    """
    Blend KG and UCB scores for robust exploration-exploitation balance.
    
    KG provides principled expected improvement of the global minimum.
    UCB provides additional exploration in high-uncertainty regions.
    
    We normalize both to [0, 1] range before blending to ensure
    comparable scales.
    """
    kg = _kg_score(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, f_best, beta=2.0)
    
    # Normalize KG to [0, 1]
    kg_min, kg_max = np.min(kg), np.max(kg)
    if kg_max - kg_min > 1e-12:
        kg_norm = (kg - kg_min) / (kg_max - kg_min)
    else:
        kg_norm = np.zeros_like(kg)
    
    # Normalize UCB to [0, 1]
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: 70% KG, 30% UCB
    # KG is the primary signal for global minimum tracking
    # UCB adds exploration in uncertain regions
    blended = 0.7 * kg_norm + 0.3 * ucb_norm
    
    return blended


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a blend of Knowledge Gradient (KG) and UCB for robust
    exploration-exploitation balance on multi-modal landscapes.
    
    KG estimates the expected reduction in the best observed value,
    providing a principled measure of how much evaluating at x is
    expected to improve our knowledge of the global minimum.
    
    UCB adds an exploration bonus for high-uncertainty regions,
    helping to discover Branin's multiple local minima.
    
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: best (lowest) observed value so far
    
    Returns: array of shape (2000,) with scores (higher = more promising)
    """
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline
    if _check_deadline():
        # Return KG scores as fallback (simpler, faster)
        return _kg_score(mu, sigma, f_best)
    
    # Use blended KG + UCB
    scores = _blend_scores(mu, sigma, f_best)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
