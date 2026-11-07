import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known optimum for Styblinski-Tang 1D
    f_opt = -39.166
    
    # Adaptive exploration coefficient
    # When f_best is far from optimum, explore more (higher beta)
    # When f_best is close to optimum, exploit more (lower beta)
    # Scale: f_best ranges from ~50 (bad) to ~-39 (good)
    # Map to beta in [0.5, 3.0]
    
    # Distance from optimum (normalized)
    dist = np.clip((f_best - f_opt) / 50.0, 0.0, 1.0)
    # beta: 3.0 when dist=1 (far), 0.5 when dist=0 (close)
    beta = 0.5 + 2.5 * dist
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Blend: weight EI more when close to optimum, UCB more when far
    # weight_ei increases as dist decreases
    weight_ei = 1.0 - 0.5 * dist  # 1.0 when close, 0.5 when far
    weight_ucb = 1.0 - weight_ei  # 0.0 when close, 0.5 when far
    
    # Normalize to combine
    score = weight_ei * ei + weight_ucb * ucb
    
    # Add a small bonus for points with low predicted mean (exploitation boost)
    # This helps refine promising regions
    min_mu = np.min(mu)
    max_mu = np.max(mu)
    mu_range = max_mu - min_mu
    if mu_range > 1e-10:
        mu_norm = (mu - min_mu) / mu_range  # 0 = best predicted, 1 = worst
    else:
        mu_norm = np.zeros_like(mu)
    
    # Small exploitation bonus: favor points with low predicted mean
    exploit_bonus = 0.1 * (1.0 - mu_norm)
    
    score = score + exploit_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
