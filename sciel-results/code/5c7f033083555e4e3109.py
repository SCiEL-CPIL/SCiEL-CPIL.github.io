import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a probabilistic expected improvement (EI) variant with a dynamic
    improvement threshold scaled by the current best value to balance
    exploration and exploitation.
    
    Args:
        mu: Mean of the GP posterior (shape: N,)
        sigma: Std of the GP posterior (shape: N,)
        f_best: Best observed function value (scalar)
        
    Returns:
        Scores for each candidate (shape: N,), higher = more promising.
    """
    # Deadline guard: track elapsed time; return best fallback if near limit
    start = time.time()
    DEADLINE = 290.0  # 40s safety margin below 330s kill
    
    # Ensure shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid log(0) and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Dynamic improvement threshold: scale by |f_best| to adapt to landscape
    # When f_best is large (early), threshold is large → more exploration
    # When f_best is small (late), threshold is small → more exploitation
    abs_fb = abs(f_best) if abs(f_best) > 1e-12 else 1e-12
    xi = 0.1 * abs_fb  # improvement threshold
    
    # Standardized EI for minimization:
    # E[improvement] = E[max(f_best - f(x), 0)]
    # = (f_best - mu + xi) * Phi((f_best - mu + xi)/sigma) + sigma * phi((f_best - mu + xi)/sigma)
    # where the argument z = (f_best - mu + xi) / sigma
    
    # Check deadline before heavy computation
    if time.time() - start > DEADLINE:
        # Fallback: return UCB-like score
        return -mu + 2.0 * sigma
    
    # Compute z-scores
    z = (f_best - mu + xi) / sigma_safe
    
    # Clip z to avoid numerical overflow in norm.pdf/cdf
    z = np.clip(z, -50, 50)
    
    # Expected improvement (for minimization)
    # EI = (f_best - mu + xi) * Phi(z) + sigma * phi(z)
    # But we want improvement over f_best, so:
    # Actually, for minimization with threshold xi:
    # EI = E[max(f_best - xi - f(x), 0)]
    # Let d = f_best - xi - mu
    # EI = d * Phi(d/sigma) + sigma * phi(d/sigma)
    d = f_best - xi - mu
    z_d = d / sigma_safe
    z_d = np.clip(z_d, -50, 50)
    
    cdf = norm.cdf(z_d)
    pdf = norm.pdf(z_d)
    
    ei = d * cdf + sigma_safe * pdf
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() - start > DEADLINE:
        return ei
    
    # Add a small UCB component for additional exploration pressure
    # The blend helps escape local minima in the rugged Weierstrass landscape
    ucb_score = -mu + 1.5 * sigma
    
    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-12 else 1.0
    ucb_min = np.min(ucb_score)
    ucb_range = ucb_max - ucb_min if ucb_range > 1e-12 else 1.0
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weight the blend based on how close we are to convergence
    # Early on (f_best far from 0), weight exploration more
    # Late on (f_best close to 0), weight exploitation more
    if abs_fb > 1e-3:
        w_ei = 0.6  # more exploration early
    elif abs_fb > 1e-5:
        w_ei = 0.7
    else:
        w_ei = 0.8  # very close to optimum, exploit more
    
    scores = w_ei * ei_norm + (1 - w_ei) * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() - start > DEADLINE:
        return scores
    
    return scores
