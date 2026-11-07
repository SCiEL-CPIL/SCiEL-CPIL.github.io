"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Optimistic Lower Confidence Bound (LCB) via Multi-Sample Thompson Sampling.
    
    Strategy: 
    1. Draw multiple samples (e.g., 50) from the GP posterior y ~ N(mu, sigma^2).
    2. For each candidate point, compute the minimum value across these samples (optimistic prediction).
    3. Subtract a scaled variance term to encourage exploration in high-uncertainty regions.
    4. This creates a deterministic score that is robust to the single-shot randomness of standard TS.
    
    The "Optimistic" part (taking the min of samples) acts like a strong exploitation signal 
    for the most likely good candidates, while the variance penalty ensures we don't get stuck 
    in local minima by forcing exploration where the GP is uncertain.
    """
    # Time guard: if we've already spent too much time, fall back to a simple deterministic score
    # to avoid timeout. However, this function is called per step, so the main cost is here.
    # We assume the caller handles the total budget, but we keep this light.
    
    # Number of samples for robust estimation
    n_samples = 50
    
    # Reshape for broadcasting: (n_samples, 2000)
    # mu shape: (2000,) -> (1, 2000)
    # sigma shape: (2000,) -> (1, 2000)
    mu_exp = mu[None, :]
    sigma_exp = sigma[None, :]
    
    # Generate random normal variables: (n_samples, 2000)
    # Use a fixed seed for reproducibility within the function call, 
    # but since mu/sigma change, the actual samples will differ across steps.
    # Using a global RandomState or default np.random is fine, but for safety 
    # and to avoid side effects, we can use a local one seeded by a constant 
    # or just rely on the fact that mu/sigma provide the variation.
    # Actually, using a fixed seed here means the *pattern* of noise is the same,
    # but scaled by sigma and shifted by mu. This is deterministic for a given (mu, sigma).
    rs = np.random.RandomState(123)
    z = rs.normal(0, 1, size=(n_samples, len(mu)))
    
    # Compute samples: y = mu + sigma * z
    # Shape: (n_samples, 2000)
    y_samples = mu_exp + sigma_exp * z
    
    # Optimistic prediction: minimum value across samples for each candidate
    # This represents the best case scenario for each point
    y_min = np.min(y_samples, axis=0)
    
    # Exploration term: we want to add a bonus for high uncertainty.
    # In LCB for minimization: score = mu - beta * sigma
    # Here, we use the sample variance or just the sigma itself as a proxy.
    # Using sigma directly is standard for LCB.
    # Let's use a dynamic beta. As f_best gets closer to 0, we can reduce exploration?
    # Or keep it constant. Beale is smooth, so constant beta is often fine.
    # A value around 1.0 to 2.0 is typical. Let's try 1.5.
    beta = 1.5
    
    # Score: Lower y_min is better (exploitation). Higher sigma is better (exploration).
    # We want to MINIMIZE y, so we want LOW scores to be good? 
    # Wait, the task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So we need to NEGATE the LCB value.
    # LCB value = y_min - beta * sigma  (this is the predicted minimum value)
    # We want to pick the point with the LOWEST LCB value.
    # Therefore, the score should be -LCB_value.
    
    lcb_value = y_min - beta * sigma
    score = -lcb_value
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
# EVOLVE-BLOCK-END
