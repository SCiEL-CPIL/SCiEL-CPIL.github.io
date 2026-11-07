import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Handle sigma == 0
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We use a fixed beta for the base LCB, but the weight will adapt.
    beta_lcb = 2.0
    lcb = mu - beta_lcb * sigma
    
    # 3. Determine the blend weight
    # We want to exploit (EI) when the model predictions (mu) have high variance (we are mapping the landscape)
    # and explore (LCB) when the model is uncertain (sigma) relative to the prediction variance.
    
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma)
    
    # If mu_std is very small, the GP is confident in the shape, so we should rely on sigma for exploration.
    # If mu_std is large, the GP is still learning the shape, so EI (which targets low mu) is good.
    
    # Weight for EI (exploitation)
    # w_ei is high when mu_std is large relative to sigma_mean.
    # w_ei = mu_std / (mu_std + sigma_mean + eps)
    eps = 1e-6
    w_ei = mu_std / (mu_std + sigma_mean + eps)
    
    # Clamp w_ei to [0.2, 0.8] to avoid extreme behavior
    w_ei = np.clip(w_ei, 0.2, 0.8)
    
    # 4. Normalize EI and LCB to [0, 1]
    # This allows us to combine them linearly.
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_norm = (ei - ei_min) / (ei_max - ei_min + eps)
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min + eps)
    
    # 5. Combine
    # Higher score is better.
    # EI: higher is better (more improvement expected).
    # LCB: lower is better (lower value predicted). 
    # Wait, LCB = mu - beta*sigma. We want to MINIMIZE the function.
    # So a LOWER LCB score is better.
    # But our acquisition function returns "higher meaning evaluate next".
    # So we should INVERT the LCB score or use -LCB.
    # Let's use -LCB for the "score" direction.
    # -LCB = -mu + beta*sigma.
    # High -LCB means low mu and/or high sigma. This is good for minimization.
    
    lcb_score = -lcb
    lcb_score_min = np.min(lcb_score)
    lcb_score_max = np.max(lcb_score)
    lcb_score_norm = (lcb_score - lcb_score_min) / (lcb_score_max - lcb_score_min + eps)
    
    # Final Score
    scores = w_ei * ei_norm + (1 - w_ei) * lcb_score_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
