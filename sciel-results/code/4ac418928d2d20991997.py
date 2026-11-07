import numpy as np
from scipy.special import erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Combines Expected Improvement (EI) with an adaptive UCB component.
    Uses a Thompson-sampling-inspired randomized exploration term that
    scales with uncertainty, providing stochastic diversity while maintaining
    the deterministic exploitation signal from EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
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
    
    # Adaptive exploration weight based on f_best
    # The Weierstrass function on [-0.5, 0.5] has values in roughly [0, 0.5]
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]
    
    # Beta for UCB: higher when far from optimum (f_best large)
    beta = 0.5 + 2.5 * f_norm
    
    # UCB score for minimization: -mu + beta * sigma
    UCB = -mu + beta * sigma_safe
    
    # Thompson-sampling-inspired randomized exploration:
    # Add a small random perturbation scaled by sigma to break ties
    # and encourage diversity in selection. This is lighter than full
    # Thompson sampling but provides similar stochastic exploration.
    # The perturbation is small enough not to override the EI signal
    # but large enough to prevent deterministic cycling.
    rng = np.random.default_rng()
    noise_scale = 0.1 * f_norm  # Scale noise with exploration need
    ts_perturbation = noise_scale * sigma_safe * rng.standard_normal(n)
    
    # Combine EI and UCB with adaptive weighting
    # EI provides the exploitation signal
    # UCB provides structured exploration
    # TS perturbation provides stochastic diversity
    w_ucb = 0.3 * f_norm
    
    # Final score: EI + weighted UCB + TS perturbation
    score = EI + w_ucb * UCB + ts_perturbation
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
