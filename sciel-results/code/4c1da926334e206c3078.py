import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the "potential" of being lower than f_best.
    # Score proportional to (beta * sigma - mu) / cost
    # We use a dynamic beta based on the uncertainty of the best point.
    # f_best is the lowest observed value. We assume mu[0] or min(mu) is a proxy for current best estimate, 
    # but f_best is the raw best.
    # Let's use a fixed beta that is slightly adaptive or just a good constant.
    # Previous best score was 0.8986 with beta=2.0.
    # Let's try a slightly higher beta to encourage exploration if exploitation is stuck, 
    # or stick to a robust value.
    # Also, ensure we handle the cost correctly.
    
    # To improve on 0.8986, we might try a different beta or a more sophisticated cost-aware metric.
    # A common improvement is to use the Expected Improvement (EI) scaled by cost, 
    # or Probability of Improvement (PI).
    # PI = P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    # EI = (f_best - mu) * norm.cdf((f_best - mu) / sigma) + sigma * norm.pdf((f_best - mu) / sigma)
    
    # Let's try EI / cost. EI is generally more effective than LCB for minimization when we have a clear f_best.
    # However, EI can be sensitive. Let's try a hybrid or just EI.
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Handle numerical issues where EI might be negative due to precision
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is 1 + mean(x_scaled). Lower cost is better.
    safe_cost = np.maximum(cost, 1e-8)
    
    scores = ei / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
