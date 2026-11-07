import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling heuristic for minimization with cost awareness.
    # We approximate the posterior sample by sampling from N(mu, sigma^2).
    # To make it deterministic and robust, we use a fixed "random" offset derived from the index
    # or a simple heuristic. However, true TS requires randomness. 
    # Since we cannot seed the RNG consistently across calls in a way that mimics 
    # "one sample per candidate" without state, we use a deterministic 
    # "expected improvement" style but with a different structure: 
    # We use the probability of improvement (PoI) but weight it by cost efficiency.
    
    # Actually, a better structural change from LCB is to use 
    # Expected Improvement (EI) for minimization, but adjusted for cost.
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # This is a different mathematical family than LCB (which is linear in mu, sigma).
    # LCB is aggressive exploration. EI is more balanced.
    
    # Let's implement Cost-Adjusted EI.
    # EI_min = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma
    # If sigma is 0, EI is max(0, f_best - mu).
    
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=-np.inf)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_safe = np.maximum(sigma_safe, 1e-6) # Avoid div by zero
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu_safe) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Using scipy.stats.norm if available, else approximate
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # If mu > f_best, improvement is 0 (since we want to minimize, lower is better)
    # Wait, EI for minimization:
    # Improvement I(x) = max(0, f_best - f(x))
    # E[I(x)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Note: If mu is very low (good), f_best - mu is positive.
    # If mu is high (bad), f_best - mu is negative.
    # The formula holds.
    
    ei = (f_best - mu_safe) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score: Higher EI per unit cost is better.
    # But we also want to ensure we don't just pick the cheapest point with tiny EI.
    # A common approach is EI / cost.
    # However, if cost is very low, EI/cost can be huge.
    # Let's use a softmax-like or normalized score.
    # Simple division is a different structural choice than the linear LCB.
    
    scores = ei / cost_safe
    
    # Handle potential inf/nan
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
