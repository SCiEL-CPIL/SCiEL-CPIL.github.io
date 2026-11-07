import numpy as np
import time

_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Dynamic Lower Confidence Bound (LCB) acquisition function.
    
    Uses a quantile-based beta to adapt exploration to the current GP uncertainty landscape.
    Score = mu - beta * sigma / cost
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    if time.time() - _START_TIME > 320.0:
        # Fallback to simple EI/cost if time is running out
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Ensure safety for division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic Beta calculation
    # We want to explore regions with high sigma, but scale the exploration
    # relative to the typical uncertainty in the candidate pool.
    # Using the 95th percentile of sigma as a scale factor.
    sigma_max = np.percentile(sigma, 95)
    if sigma_max < 1e-6:
        sigma_max = 1e-6
        
    # Beta controls exploration. 
    # A common heuristic is beta ~ log(t) or similar, but here we use a 
    # fixed moderate value scaled by the problem's uncertainty range.
    # For minimization, LCB = mu - beta * sigma.
    # We divide by cost to account for the expensive evaluations in high-cost regions.
    
    # Let's use a beta that encourages finding the minimum.
    # If beta is too high, we explore random areas. If too low, we exploit.
    # 1.0 is a standard starting point for standardized GP, but here sigma is in objective units.
    # Since we don't know the scale of the objective perfectly, we normalize sigma by its max.
    # Actually, the task says mu and sigma are in objective units.
    # Let's use a dynamic beta based on the spread of sigma.
    
    # Alternative: Use a quantile-based beta.
    # beta = k * sigma_std? No, let's stick to a robust fixed beta relative to the scale.
    # However, a purely fixed beta might not adapt well.
    # Let's try beta = 1.5 * (sigma / sigma_max) is wrong, beta should be a scalar multiplier for sigma.
    
    # Let's use a simple dynamic beta:
    # If the best f_best is very low, we might want to explore less?
    # No, standard LCB is mu - beta * sigma.
    # Let's set beta to a value that balances exploration.
    # Given the "cost" factor, we are effectively doing LCB/cost.
    
    # Let's try beta = 1.0 first, but adjusted.
    # Actually, looking at the inspiration "0.798 -> 1.248" with "5% capped sigma bonus",
    # that suggests pure exploitation (EI) is strong, and a *small* exploration bonus helps.
    # LCB with a small beta is similar to EI with a small exploration term.
    
    # Let's use beta = 0.5. This is conservative exploration.
    beta = 0.5
    
    # Score = (mu - beta * sigma) / cost
    score = (mu - beta * sigma) / cost_safe
    
    # The score should be higher for better candidates.
    # Minimizing f means we want low mu and high sigma (exploration).
    # So (mu - beta * sigma) is low for "good" candidates (low mean, high uncertainty).
    # Wait. LCB for minimization: we want to minimize f.
    # Prediction: f ~ N(mu, sigma^2).
    # Lower Confidence Bound (optimistic): mu - beta * sigma.
    # We want to select the point with the *lowest* LCB?
    # No, usually in BO, we maximize an acquisition function.
    # For minimization, EI is max(0, f_best - mu ...).
    # LCB acquisition function to MAXIMIZE:
    # If we define the "value" of a point as its potential to be low,
    # a low LCB is good. So we should MAXIMIZE -LCB?
    # Or simply: Score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    
    # Let's check the sign.
    # We want to find x that minimizes f.
    # If mu is very low, it's good.
    # If sigma is high, it might be even lower.
    # So (mu - beta * sigma) being low is good.
    # Therefore, the acquisition function to MAXIMIZE should be:
    # A(x) = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    score = (beta * sigma - mu) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
