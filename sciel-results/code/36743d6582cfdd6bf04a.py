import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI is defined for maximization usually, but here we minimize.
    # Improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to maximize the potential for a lower value.
    # Lower bound of the function is mu - beta * sigma.
    # We maximize the negative of the lower bound? No.
    # We want to find points where the lower bound is low.
    # So we maximize (mu - beta * sigma) ? No, that favors high mu.
    # We want to maximize the "optimism" about how low f(x) can be.
    # Optimistic lower bound: L(x) = mu(x) - beta * sigma(x).
    # We want to maximize L(x) ? No, L(x) is a lower estimate.
    # If L(x) is -1.0 and L(y) is -0.5, x is better (lower).
    # So we want to maximize -L(x) = -mu(x) + beta * sigma(x).
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Cost-aware scores
    # cost is given as array
    score_ei = ei / cost
    score_ucb = ucb / cost
    
    # 4. Z-score normalization to balance scales
    # This allows the exploration term (UCB) to compete with exploitation (EI)
    # even if their raw magnitudes differ significantly.
    
    def zscore(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-9:
            return np.zeros_like(arr)
        return (arr - m) / s
        
    z_ei = zscore(score_ei)
    z_ucb = zscore(score_ucb)
    
    # 5. Combine
    # 75% weight to exploitation (EI), 25% to exploration (UCB)
    final_score = 0.75 * z_ei + 0.25 * z_ucb
    
    return final_score
