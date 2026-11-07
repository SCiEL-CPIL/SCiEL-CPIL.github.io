import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in CDF/PDF
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Quantile of Improvement (QI): conditional expected improvement given improvement
    # QI = E[f_best - f | f < f_best]
    # This equals: (f_best - mu) * Phi(z) + sigma * phi(z) / Phi(z)
    # But we need to handle the case where Phi(z) is very small
    
    # Expected Improvement (EI) = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # QI conditional = EI / P(improvement) = EI / cdf
    # Use safe division
    cdf_safe = np.maximum(cdf, 1e-9)
    qi_cond = ei / cdf_safe
    
    # Cost-aware score: divide by cost to account for evaluation expense
    # Cost is in [1, 2] for this problem
    scores = qi_cond / cost
    
    # Add a small exploration bonus proportional to sigma to encourage exploring uncertain regions
    # This helps in multimodal landscapes like Rastrigin
    exploration_bonus = 0.1 * sigma_safe / cost
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
