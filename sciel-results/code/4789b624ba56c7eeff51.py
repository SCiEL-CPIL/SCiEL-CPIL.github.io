import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Cost-aware scoring:
    # The cost function is 1 + mean(x) where x is scaled to [0,1].
    # So cost ranges from 1 to 2.
    # We want to balance expected improvement against the cost of evaluation.
    # A common approach is to use EI / cost or EI - lambda * cost.
    # However, the previous version used (EI + 0.5 * PoI) / cost.
    # Let's try a slightly different blend to see if it improves.
    # Perhaps we should weight the PoI term more heavily when sigma is large?
    # Or maybe we should use a UCB-like term for exploration.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Let's try: Score = EI / cost + beta * PoI / cost
    # But also consider that high sigma regions might be worth exploring more.
    # Let's add a small UCB component.
    
    # UCB for minimization: mu - kappa * sigma
    # But we want to combine with EI.
    
    # Let's stick to the EI/PoI blend but adjust the weights.
    # The previous score was 1.2255.
    # Let's try increasing the weight of PoI slightly or using a different cost exponent.
    
    # Option 1: EI / cost^1.5 (penalize high cost more)
    # Option 2: (EI + PoI) / cost
    # Option 3: EI / cost + 0.1 * sigma / cost (explicit exploration bonus)
    
    # Let's try Option 3: EI / cost + 0.1 * sigma / cost
    # This explicitly rewards high uncertainty (sigma) independent of PoI.
    
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    scores = ei / cost_safe + exploration_bonus
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
