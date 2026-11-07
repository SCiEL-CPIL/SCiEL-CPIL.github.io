import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Lower Confidence Bound
    # We want to minimize f.
    # Lower bound on f is mu - kappa * sigma.
    # Potential improvement is f_best - (mu - kappa * sigma).
    # We divide by cost to favor cheaper evaluations.
    
    kappa = 1.5
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    lcb = mu - kappa * sigma
    improvement = f_best - lcb
    
    # To avoid negative scores dominating (which happen if LCB > f_best),
    # we can clip or just let them be negative.
    # Usually, acquisition functions are shifted to be positive, but relative order is what matters.
    # However, if all scores are negative, the "max" will pick the least negative one.
    # This is fine.
    
    scores = improvement / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
