import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost.
    # This is the core strategy from the highest-scoring inspiration (1.0000).
    # It balances exploitation (high EI) against cost (low cost).
    
    # Ensure sigma is positive to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for the standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Add an exploration bonus to encourage sampling in high-uncertainty regions
    # This helps escape local minima in the multimodal Levy landscape
    exploration_bonus = 0.5 * sigma_safe * pdf_z
    
    # Cost-aware score: divide by cost
    # This encourages evaluating points that are both promising (high EI) and cheap
    scores = (ei + exploration_bonus) / cost
    
    # Handle any non-finite values to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
