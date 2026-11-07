import numpy as np
from scipy.special import erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement (EI) with an adaptive exploration bonus
    that scales with the ratio of sigma to the expected improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    sqrt_2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Phi(z) = 0.5 * erfc(-z/sqrt(2))
    Phi = 0.5 * erfc(-z_clipped / sqrt_2)
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    phi = np.exp(-0.5 * z_clipped**2) / sqrt_2pi
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Compute a UCB-like term for exploration
    # Use an adaptive beta that is higher when f_best is large (far from optimum)
    # and lower when f_best is small (close to optimum)
    # The Weierstrass function on [-0.5, 0.5] has values in roughly [0, 0.5]
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]
    
    # beta ranges from ~0.5 (exploitative) to ~3.0 (exploratory)
    beta = 0.5 + 2.5 * f_norm
    
    UCB = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI already has both exploitation and exploration properties
    # Add a weighted UCB term to encourage more exploration in early stages
    # The weight on UCB should be higher when f_best is large
    w_ucb = 0.3 * f_norm
    
    # Normalize both to comparable scales
    ei_mean = np.mean(EI)
    ucb_mean = np.mean(UCB)
    ei_std = np.std(EI) + 1e-12
    ucb_std = np.std(UCB) + 1e-12
    
    EI_norm = (EI - ei_mean) / ei_std
    UCB_norm = (UCB - ucb_mean) / ucb_std
    
    # Final score: primarily EI, with a UCB component for exploration
    score = EI + w_ucb * UCB
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
