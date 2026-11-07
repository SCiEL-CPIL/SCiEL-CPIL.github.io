import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and 
    Lower Confidence Bound (LCB) via a max operation.
    
    EI focuses on points likely to improve the current best.
    LCB focuses on points with high uncertainty or low predicted values.
    Taking the max ensures we don't ignore strong exploratory candidates 
    early on, nor strong exploitative candidates later.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # --- Expected Improvement (EI) ---
    # Using xi=0 for pure EI, which is the Knowledge Gradient for 
    # single-step regret reduction.
    z_ei = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z_ei)
    Phi_z = norm.cdf(z_ei)
    
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # --- Lower Confidence Bound (LCB) ---
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma
    # Beta decays to shift from exploration to exploitation.
    # We estimate progress based on f_best. 
    # Min value is approx -39.16. Max initial values can be ~100.
    # Let's use a simple decay based on the number of evaluations if we had it, 
    # but we don't. We use f_best as a proxy.
    
    # If f_best is close to the global min, we should exploit more (lower beta).
    # If f_best is high, we should explore more (higher beta).
    f_min_global = -39.1662
    # Normalize regret
    regret = f_best - f_min_global
    # Assume max regret is around 100 (typical for ST 1D scaled or 20D)
    # Actually, for 1D ST, max is ~5, min is ~-0.18.
    # But the prompt says min is -39.16. This implies scaling.
    # Let's assume a range of [f_min_global, f_min_global + 100] for normalization.
    max_regret = 100.0 
    progress = np.clip(1.0 - (regret / max_regret), 0.0, 1.0)
    
    # Beta starts at 3.0 (explore) and decays to 0.5 (exploit)
    beta = 0.5 + 2.5 * (1.0 - progress)
    
    lcb_scores = -mu + beta * sigma_safe
    
    # --- Combine ---
    # Use the maximum of the two scores to ensure we consider both 
    # the best expected improvement and the most optimistic bound.
    scores = np.maximum(ei_scores, lcb_scores)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
