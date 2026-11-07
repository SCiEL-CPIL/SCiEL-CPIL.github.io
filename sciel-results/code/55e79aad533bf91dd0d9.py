import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Strategy:
    1. Compute the Expected Improvement (EI) of each candidate point over f_best.
       EI(x) = E[max(f_best - f(x), 0)]
       For a Gaussian posterior N(mu, sigma^2), this is:
       EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma, Phi is the CDF, and phi is the PDF.
       
    2. Incorporate the cost of evaluation. The cost is given as a scalar per point
       (representing the marginal cost of evaluating that specific point).
       We subtract a weighted cost from the EI to get a net expected gain per unit cost.
       To make the cost penalty meaningful relative to the objective scale, we scale
       the cost. The cost is in [1, 2]. The objective improvement is typically small.
       A simple linear penalty: Score = EI - lambda * cost.
       
    3. To handle the case where sigma is very small (exploited points), EI becomes 0
       if mu > f_best, and (f_best - mu) if mu < f_best. The formula handles this naturally.
       
    4. Return the score where higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and numerically stable
    # If sigma is 0, we treat it as a known value.
    # We add a small epsilon to avoid division by zero in the z-score calculation.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning there is a high probability of improvement.
    # If mu > f_best, z < 0, meaning there is a low probability of improvement.
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # Optimization: norm.cdf and norm.pdf can be slow for large arrays if not vectorized,
    # but scipy.stats functions are vectorized.
    # However, we can use the standard normal PDF formula for speed:
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    # Clip z to avoid extreme values causing underflow/overflow in exp, though norm.cdf is robust.
    # For very large positive z, Phi(z) -> 1, phi(z) -> 0. EI -> f_best - mu.
    # For very large negative z, Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    
    # Using scipy.stats for robustness
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment
    # The cost is in [1, 2]. 
    # We want to penalize expensive points.
    # If we just subtract cost, the scale of EI (which can be small, e.g., 0.01) 
    # might be dwarfed by the cost (1.0 - 2.0).
    # We need a scaling factor lambda.
    # Let's analyze the typical scale of EI. 
    # In early stages, EI is larger. In late stages, EI is small.
    # A common approach is to divide EI by cost, or subtract a small fraction of cost.
    # Given the "Cost-aware" requirement, let's use a weighted sum.
    # Score = EI - lambda * cost
    # If lambda is too high, we always pick the cheapest point (lowest cost).
    # If lambda is 0, we ignore cost.
    # Let's try a small lambda, e.g., 0.05.
    # Alternatively, we can normalize cost to [0, 1] first.
    # cost_norm = (cost - min_cost) / (max_cost - min_cost + eps)
    # But min/max might change or be known. The problem states cost is 1 + mean(x_scaled).
    # So cost is in [1, 2].
    
    # Let's use a simple linear penalty.
    # To make the penalty adaptive, we could scale it by the current best improvement potential,
    # but that's complex. Let's stick to a fixed lambda for stability.
    # Based on typical BO practices, the cost penalty should be comparable to the expected improvement.
    # If f_best is around 0.1 and we expect 0.01 improvement, cost 1.5 is huge.
    # So we need a very small lambda.
    # Let's try lambda = 0.01.
    
    lambda_cost = 0.01
    score = ei - lambda_cost * cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
