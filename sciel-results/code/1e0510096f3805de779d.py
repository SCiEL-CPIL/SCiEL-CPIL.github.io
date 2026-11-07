import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB Acquisition Function.
    Adjusts the exploration parameter beta based on the current uncertainty landscape.
    High uncertainty (early) -> High Beta (Explore).
    Low uncertainty (late) -> Low Beta (Exploit).
    """
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate statistics to determine 'maturity'
    sigma_min = np.min(sigma)
    sigma_mean = np.mean(sigma)
    
    # Avoid division by zero
    if sigma_mean < 1e-9:
        beta = 1.0
    else:
        # Ratio of min to mean uncertainty
        # If min is close to mean, uncertainty is uniform (early stage) -> r ~ 1
        # If min is much smaller than mean, uncertainty is localized (late stage) -> r ~ 0
        r = sigma_min / sigma_mean
        # Clip r to [0, 1] just in case of numerical issues
        r = np.clip(r, 0.0, 1.0)
        
        # Map r to beta
        # beta in [0.5, 3.0]
        beta = 0.5 + 2.5 * r
    
    # UCB score for minimization:
    # We want to minimize f.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to MAXIMIZE the score returned.
    # So we return -LCB(x) = beta * sigma(x) - mu(x)
    scores = beta * sigma - mu
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
