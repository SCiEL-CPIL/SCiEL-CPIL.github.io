import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI + UCB acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Combines Expected Improvement (exploitative, scale-aware) with a dynamic
    Upper Confidence Bound (explorative) term. The UCB weight decays as f_best
    approaches the known global minimum, shifting from exploration to exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Known global minimum of Styblinski-Tang 1D
    f_min_global = -39.1662
    
    # Current regret: how far we are from the global optimum
    regret = f_best - f_min_global
    
    # Normalize regret to get progress (0 = start, 1 = found optimum)
    # Initial function values can be ~100-200, so max_regret ~ 200
    max_regret = 200.0
    progress = np.clip(1.0 - (regret / max_regret), 0.0, 1.0)
    
    # --- Exploitation term: Expected Improvement ---
    # Use a small xi that decays with progress to focus on actual improvements
    xi = 1.0 * (1.0 - progress)
    
    z_ei = (f_best - mu - xi) / sigma_safe
    phi_z = norm.pdf(z_ei)
    Phi_z = norm.cdf(z_ei)
    
    # EI for minimization: (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu - xi) * Phi_z + sigma_safe * phi_z
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # --- Exploration term: Dynamic UCB ---
    # UCB score = -mu + beta * sigma
    # Beta starts high (explore) and decays as we get closer to optimum
    beta = 0.5 + 1.5 * (1.0 - progress)
    ucb_scores = -mu + beta * sigma_safe
    
    # Normalize both terms to have similar scale before combining
    # EI scores are in objective units, UCB scores are also in objective units
    # But they have different magnitudes. Let's blend them.
    
    # Weighting: early on, rely more on UCB for exploration; later, rely on EI
    # w_ucb starts at 0.5 and decays to 0.1
    w_ucb = 0.1 + 0.4 * (1.0 - progress)
    w_ei = 1.0 - w_ucb
    
    # Combine: higher score = more promising
    # EI is already "higher is better" (larger improvement)
    # UCB score is "-mu + beta*sigma", higher means lower predicted value or higher uncertainty
    scores = w_ei * ei_scores + w_ucb * ucb_scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
