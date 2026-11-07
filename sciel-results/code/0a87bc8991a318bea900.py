import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safe sigma
    sigma_safe = sigma + 1e-9
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # PI = Phi((f_best - mu) / sigma)
    # This naturally saturates at 1.0 for certain improvements and decays smoothly
    pi = norm.cdf(z)
    
    # Cost-aware weighting using exponentiation
    # PI^(1/cost) prioritizes cheap, high-confidence improvements
    # When cost is low, the exponent is high, so PI must be close to 1 to get a high score
    # When cost is high, the exponent is low, so even moderate PI can get a decent score
    # But we want to penalize high cost, so we use cost in the denominator of the exponent
    # Actually, let's use: score = PI / (cost ** beta)
    # Or: score = PI ** (1/cost)
    # Let's try: score = PI * (1 / cost ** beta)
    # The directive suggests: PI^(1/cost)
    
    # To avoid numerical issues with 0^0 or very small numbers, clip cost
    cost_safe = np.maximum(cost, 1e-6)
    
    # Use the multiplicative PI approach with cost penalty
    # score = PI ** (1/cost)
    # This means if cost=1, score=PI; if cost=2, score=PI^0.5; if cost=4, score=PI^0.25
    # This naturally balances exploration (high PI from uncertainty) with exploitation (low cost)
    
    score = np.power(pi, 1.0 / cost_safe)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
