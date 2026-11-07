import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB)
    # Beta is a hyperparameter for exploration. 2.0 is a common default.
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # 3. Incorporate Cost
    # The prompt suggests combining EI/cost and UCB/cost.
    # Cost is given as an array.
    ei_cost = ei / cost
    ucb_cost = ucb / cost
    
    # 4. Z-score normalization to make them comparable
    # We calculate the mean and std of the scores across the 2000 candidates
    # to normalize them to zero mean and unit variance.
    
    def zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-9:
            return arr - mean
        return (arr - mean) / std
        
    z_ei = zscore(ei_cost)
    z_ucb = zscore(ucb_cost)
    
    # 5. Combine with weights
    # The hint suggests 25% exploration weight.
    # So 75% Exploitation (EI) and 25% Exploration (UCB).
    w_ei = 0.75
    w_ucb = 0.25
    
    scores = w_ei * z_ei + w_ucb * z_ucb
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
