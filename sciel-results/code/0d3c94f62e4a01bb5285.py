# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid approach with an adaptive exploration-exploitation balance.
    Uses Expected Improvement (EI) as the base, but adds a significant exploration term 
    (like LCB) scaled by the local uncertainty, weighted by a global factor that 
    decreases as the maximum uncertainty in the domain decreases.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) score
    # We use a fixed beta for LCB, but scale it dynamically
    beta_base = 1.5
    lcb = -mu + beta_base * sigma
    
    # Adaptive blending
    # max_sigma indicates how much uncertainty remains globally.
    # Early on, max_sigma is high (close to 1.0 for standardized targets).
    # Later, max_sigma drops.
    max_sigma = np.max(sigma)
    
    # We want to explore more when max_sigma is high.
    # Let's create a weight for exploration.
    # If max_sigma > 0.8, weight_explore is high.
    # If max_sigma < 0.2, weight_explore is low.
    
    # A simple mapping:
    # weight_explore = 1.0 if max_sigma > 0.8
    # weight_explore = 0.0 if max_sigma < 0.2
    # Linear interpolation in between? Or smoothstep?
    
    # Let's try a more aggressive exploration in the very beginning.
    # The first 5 points are random. The 6th point is the first acquisition.
    # At step 6, we have 5 points. The GP will have high uncertainty everywhere except near the 5 points.
    # max_sigma will likely be close to 1.0.
    
    # If we use pure EI early on, it might exploit the few known points too much.
    # If we use pure LCB, it might jump to random high-uncertainty areas.
    
    # A robust strategy is to use EI but add a "bonus" for high sigma.
    # Score = EI + alpha * sigma
    # Where alpha is large early on.
    
    # Let's define alpha based on max_sigma.
    # alpha = 1.0 if max_sigma > 0.7 else 0.0?
    # Let's use a smooth transition.
    
    # Transition from 0.5 to 0.9
    # If max_sigma is 0.9, alpha = 1.0
    # If max_sigma is 0.5, alpha = 0.0
    # This encourages exploration when uncertainty is still high.
    
    alpha = np.clip((max_sigma - 0.5) / 0.4, 0.0, 1.0)
    
    # The exploration term:
    # Using sigma directly is a bit crude. 
    # LCB is -mu + beta*sigma.
    # EI is (f_best - mu)Phi(z) + sigma phi(z).
    # Note that when mu is very low (good), EI is high.
    # When sigma is high, EI is also higher (due to the second term).
    # So EI already has some exploration.
    
    # However, EI is bounded and can be small.
    # Let's try a different structural change:
    # Use a "Pareto" front of EI and LCB? No, that's complex.
    
    # Let's try: Score = EI + alpha * (LCB - min(LCB))
    # This adds a scaled LCB component.
    
    lcb_min = np.min(lcb)
    lcb_scaled = lcb - lcb_min
    
    scores = ei + alpha * lcb_scaled
    
    # Time check
    if time.time() - start_time > 325:
        return ei
    
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
# EVOLVE-BLOCK-END
