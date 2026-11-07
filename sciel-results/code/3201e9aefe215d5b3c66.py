import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Adaptive UCB with Expected Improvement.
    
    The exploration parameter beta is adapted based on the current best observed value f_best.
    - When f_best is large (poor solution found), beta is high to encourage exploration.
    - When f_best is small (good solution found), beta decreases to encourage exploitation.
    
    The final score is a weighted combination of:
    1. GP-UCB score: -mu + beta * sigma (encourages exploration of high uncertainty)
    2. Expected Improvement (EI): encourages exploitation of regions likely to improve f_best
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Adaptive Beta calculation
    # We want beta to be high when f_best is high, and lower when f_best is low.
    # Beale function global min is 0. Typical starting values are ~10-100.
    # Use a logistic decay centered around f_best = 10.
    # beta_min = 1.5 (exploit), beta_max = 10.0 (explore)
    
    k = 1.0 
    center = 10.0
    beta_min = 1.5
    beta_max = 10.0
    
    # Logistic function: 1 / (1 + exp(-k * (f_best - center)))
    # If f_best >> center, sigmoid ~ 1.
    # If f_best << center, sigmoid ~ 0.
    z = k * (f_best - center)
    z = np.clip(z, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(-z))
    
    beta = beta_min + (beta_max - beta_min) * sigmoid_val
    
    # Calculate UCB score for minimization
    # We want to maximize the score.
    # LCB = mu - beta * sigma. We want to minimize LCB.
    # So we maximize -LCB = -mu + beta * sigma.
    ucb_score = -mu + beta * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    z_ei = (f_best - mu) / sigma
    # Clip z to prevent overflow in exp
    z_ei = np.clip(z_ei, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize both scores to [0, 1] for combination
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ucb_score)
    
    # Combine normalized scores
    # Weight: More weight on UCB when beta is high (early), more on EI when beta is low (late)
    # beta ranges from beta_min (1.5) to beta_max (10.0)
    # Normalize beta to [0, 1] for weighting
    beta_norm = (beta - beta_min) / (beta_max - beta_min)
    beta_norm = np.clip(beta_norm, 0.0, 1.0)
    
    # Weight for UCB: higher when beta is higher (more exploration)
    # Weight for EI: higher when beta is lower (more exploitation)
    w_ucb = 0.5 + 0.5 * beta_norm  # Range [0.5, 1.0]
    w_ei = 1.0 - w_ucb             # Range [0.0, 0.5]
    
    # Ensure weights sum to 1
    total_w = w_ucb + w_ei
    if total_w > 0:
        w_ucb /= total_w
        w_ei /= total_w
    
    # Final score
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
