import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Calculate median uncertainty for normalization
    # Use a small epsilon to avoid division by zero
    sigma_med = np.median(sigma)
    if sigma_med < 1e-9:
        sigma_med = 1e-9
        
    # Calculate the ratio of current best value to median uncertainty
    # This ratio indicates how "far" we are from the minimum relative to our uncertainty
    # High ratio: We are far from min (or uncertainty is low compared to distance), 
    # but typically in BO, if f_best is high, we haven't found the min yet.
    # If f_best is low, we are close.
    # The knowledge suggests: "shifting from exploration to exploitation as f_best approaches 0"
    # If f_best is large, we want to explore more? 
    # Actually, if f_best is large, it means the current best is bad. 
    # If sigma is large, we are uncertain.
    # Let's look at the ratio R = f_best / sigma_med.
    # If R is large, f_best is large relative to uncertainty. 
    # This might happen in early stages if the function values are large.
    # If R is small, f_best is small relative to uncertainty.
    
    # The successful strategy was: "beta normalized by current best value relative to median uncertainty"
    # Let's try: beta = beta_max * (1 - exp(-k * R)) ? 
    # Or the linear fraction: beta = beta_max * R / (1 + R)
    
    # If R is small (f_best << sigma_med), beta is small -> Exploitation.
    # If R is large (f_best >> sigma_med), beta is large -> Exploration.
    # This seems counter-intuitive? 
    # If f_best is large, we are far from the minimum. We SHOULD explore.
    # If f_best is small, we are close to the minimum. We SHOULD exploit.
    # So: Large f_best -> Large Beta (Explore). Small f_best -> Small Beta (Exploit).
    # The formula beta = beta_max * R / (1 + R) does exactly this.
    
    beta_max = 3.0
    R = f_best / sigma_med
    beta = beta_max * (R / (1.0 + R))
    
    # Calculate UCB score for minimization
    # We want to minimize mu - beta * sigma
    # So we return the negative of that for maximization
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
