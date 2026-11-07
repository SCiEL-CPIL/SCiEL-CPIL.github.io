import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-adjusted LCB: exploration term scaled by 1/cost
    # score = -mu - beta * sigma / cost
    # Higher score = better (more likely to be minimum, cheaper)
    
    # Beta controls exploration-exploitation tradeoff
    # Start with a moderate beta, could tune later
    beta = 1.0
    
    # Compute LCB with cost-adjusted exploration
    # Lower LCB is better for minimization, so we negate for "higher is better"
    lcb = mu + beta * sigma / cost
    score = -lcb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
