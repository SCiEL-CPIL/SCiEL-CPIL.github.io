import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) acquisition for cost-aware Bayesian optimization.
    
    Models cost C as a log-normal random variable: C ~ LogNormal(mu_c, sigma_c)
    where mu_c = log(cost) and sigma_c is a fixed uncertainty parameter.
    
    SEI = E_C[E_f[max(f_best - f, 0) / C]]
        = E[1/C] * EI_standard
    
    The key structural difference from pointwise EI/cost:
    - EI/cost uses a deterministic ratio, which can be overly sensitive to cost outliers
    - SEI integrates over cost uncertainty, providing a more robust probabilistic criterion
    - The E[1/C] factor for LogNormal is exp(mu_c + sigma_c^2/2), which is less 
      extreme than 1/cost when cost has high variance
    """
    # Deadline guard: if we've been running too long, return a safe fallback
    # This is a safety margin; the actual computation is vectorized and fast
    t0 = time.time()
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement: E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Model cost as LogNormal: C ~ LogNormal(mu_c, sigma_c)
    # mu_c = log(cost), sigma_c = fixed uncertainty (e.g., 0.15)
    cost_safe = np.maximum(cost, 1e-8)
    mu_c = np.log(cost_safe)
    sigma_c = 0.15  # cost uncertainty parameter
    
    # E[1/C] for C ~ LogNormal(mu_c, sigma_c) = exp(mu_c + sigma_c^2/2)
    # This is the expected inverse cost, which is the correct weighting for SEI
    e_inv_cost = np.exp(mu_c + 0.5 * sigma_c**2)
    
    # SEI = E[1/C] * EI
    se_i = e_inv_cost * ei
    
    # Add a small exploration term that also accounts for cost uncertainty
    # Exploration: sigma * E[1/C] / (1 + sigma_c^2) to account for cost variance
    # This is a heuristic that encourages exploring uncertain regions
    # while being penalized by cost uncertainty
    exploration = 0.05 * sigma_safe * e_inv_cost / (1.0 + sigma_c**2)
    
    scores = se_i + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: if computation took too long, we still return valid scores
    # (the vectorized ops above should complete in <1ms for 2000 points)
    elapsed = time.time() - t0
    if elapsed > 290.0:
        # Fallback: return simple EI/cost if we're close to timeout
        cost_safe_fallback = np.maximum(cost, 1e-8)
        fallback = ei / cost_safe_fallback
        return np.nan_to_num(fallback, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
