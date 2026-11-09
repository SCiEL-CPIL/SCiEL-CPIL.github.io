import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Normalized Expected Improvement (EI)
    and Normalized Upper Confidence Bound (UCB) for minimization.
    
    Based on findings that normalizing prevents scale dominance and a 70/30 
    blend balances exploitation and exploration effectively for multimodal 
    functions like Styblinski-Tang.
    """
    # Ensure sigma is positive to avoid division issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to maximize the Lower Confidence Bound: LCB = mu - beta * sigma
    # Note: In minimization contexts, "UCB" often refers to the bound used for selection.
    # If we define Score = mu - beta * sigma, we maximize this to find points 
    # that are potentially very low (low mu) or highly uncertain (high sigma).
    beta = 2.5  # Exploration parameter
    ucb_raw = mu - beta * sigma_safe
    
    # 3. Normalization
    # Normalize to zero mean and unit variance to prevent one metric from 
    # dominating due to scale differences as the GP evolves.
    
    ei_mean = np.mean(ei_raw)
    ei_std = np.std(ei_raw)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei_raw)
    else:
        ei_norm = (ei_raw - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb_raw)
    ucb_std = np.std(ucb_raw)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb_raw)
    else:
        ucb_norm = (ucb_raw - ucb_mean) / ucb_std
        
    # 4. Blend
    # 70% EI, 30% UCB
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    return scores
