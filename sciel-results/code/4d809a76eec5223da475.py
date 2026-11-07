import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling-inspired logic.
    
    Strategy:
    1. Instead of a fixed LCB (mu - kappa*sigma), we sample a target value 
       delta from the distribution of the global minimum. We approximate this 
       by sampling from a Normal distribution centered at f_best with 
       standard deviation proportional to the uncertainty at the best point 
       (or a global uncertainty scale).
    2. We compute the Expected Improvement (EI) of each candidate relative to 
       this sampled delta.
    3. We apply a cost-aware weighting. Since we want to minimize total cost 
       while finding the minimum, we can interpret the cost as a penalty on 
       the "value" of the point. A common way to do this in TS-like frameworks 
       is to adjust the improvement threshold or the score. 
       
       However, a more direct interpretation for "cost-aware acquisition" in 
       a budget-constrained BO is to favor points that offer high improvement 
       per unit cost. 
       
       Let's refine the "sampled shift" strategy:
       The prompt suggests: "sampling a single global shift delta from the 
       implied posterior distribution of the function minimum... and scoring 
       candidates by their deterministic expected improvement relative to this 
       sampled minimum".
       
       So:
       1. Estimate sigma_best. Since we don't have x, we can use the median 
          of sigma or the max sigma as a proxy for the scale of uncertainty. 
          A better proxy for the uncertainty at the *minimum* is difficult 
          without x. However, f_best is the minimum *observed*. 
          The uncertainty at the minimum is likely low if we've found it, 
          but high if we haven't. 
          Let's use a robust estimate of the current uncertainty level, e.g., 
          np.median(sigma) or a quantile. 
          Actually, the standard deviation of the GP at the best point is not 
          directly available. But we can assume that if f_best is far from 
          -0.8, we are in a high-uncertainty region. 
          
       Let's use a simpler, robust approach for the "sampled minimum":
       Let sigma_scale = np.median(sigma).
       Let delta ~ N(f_best, sigma_scale).
       
       Then Score(x) = EI(x | mu(x), sigma(x), delta) * Cost_Weight(x)
       
       EI(x) = (delta - mu(x)) * Phi((delta - mu(x))/sigma(x)) + sigma(x) * phi((delta - mu(x))/sigma(x))
       
       Cost_Weight(x): 
       The cost is 1 + mean(x_scaled). Lower cost is better.
       We can divide the EI by the cost, or subtract a penalty. 
       Dividing EI by cost is a "improvement per cost" metric, which aligns 
       with maximizing value under budget constraints.
       
       Score = EI(x) / cost(x)
       
       This naturally favors cheap points with high potential improvement.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Sample the target minimum delta
    # Use the median sigma as a proxy for the current uncertainty scale.
    # If the GP has converged, sigma will be small everywhere, and delta will 
    # be close to f_best. If the GP is uncertain, sigma will be large, and 
    # delta will be sampled further away, encouraging exploration.
    sigma_scale = np.median(sigma)
    
    # Add a small floor to sigma_scale to prevent zero variance in delta
    sigma_scale = max(sigma_scale, 1e-3)
    
    # Sample delta from N(f_best, sigma_scale)
    # Using a fixed seed or random? Random is part of Thompson Sampling.
    # However, for reproducibility in a single call, we just sample once.
    delta = np.random.normal(f_best, sigma_scale)
    
    # 2. Compute Expected Improvement relative to delta
    # z = (delta - mu) / sigma
    z = (delta - mu) / sigma
    
    # EI = (delta - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (delta - mu) * Phi_z + sigma * phi_z
    
    # 3. Cost-awareness
    # We want to maximize EI per unit cost.
    # Score = EI / cost
    scores = EI / cost
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() > deadline:
        return np.ones_like(mu)
        
    return scores
