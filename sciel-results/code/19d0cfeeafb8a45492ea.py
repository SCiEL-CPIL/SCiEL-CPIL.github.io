import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might make it tiny negative)
    ei = np.maximum(ei, 0.0)
    
    # Apply exponential cost penalty
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2.
    # We use lambda=1.0 as suggested by the measured improvement in the prompt.
    # Normalizing cost by subtracting 1 helps focus the penalty on the *extra* cost,
    # but the prompt's "exponential cost penalty (lambda=1)" likely implies 
    # exp(-lambda * cost) or exp(-lambda * (cost - min_cost)).
    # Given cost is in [1, 2], exp(-1*1) vs exp(-1*2) is a factor of e difference.
    # Let's stick to the direct application: EI * exp(-lambda * cost)
    # However, if cost is always >= 1, multiplying by exp(-1) is a constant factor 
    # that doesn't change ranking. So we should normalize.
    # Let's use exp(-lambda * (cost - 1.0)) to make the minimum cost have penalty 1.
    
    lambda_penalty = 1.0
    penalty = np.exp(-lambda_penalty * (cost - 1.0))
    
    scores = ei * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
