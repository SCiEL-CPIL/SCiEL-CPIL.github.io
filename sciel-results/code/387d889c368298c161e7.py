import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we finish well within 300s.
    # Since this is a single call, we just make sure computations are fast.
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-10)
    
    # Beale function global minimum is 0 at (3,0).
    # Domain is [-4.5, 4.5]^2.
    # We don't have coordinates, so we rely on mu and sigma.
    
    # Strategy: 
    # The previous program used a combination of UCB and EI.
    # The inspiration with score 6.0973 likely used a better acquisition function.
    # Common strong acquisition functions for BO:
    # 1. Expected Improvement (EI)
    # 2. Probability of Improvement (PI)
    # 3. Upper Confidence Bound (UCB)
    # 4. Knowledge Gradient (KG) - expensive
    # 5. Entropy Search - expensive
    
    # For 2000 candidates, EI and UCB are fast.
    # Let's try a pure EI first, as it is often very effective for minimization.
    # Or a tuned UCB.
    
    # Let's look at the "inspiration 1" which scored 6.0973.
    # It's described as "The task is to implement an acquisition function...".
    # It uses numpy, scipy.
    # Without seeing the code, I can guess it might be a well-tuned EI or UCB.
    
    # Let's try Expected Improvement (EI) for minimization.
    # Improvement I = max(f_best - y, 0)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # However, EI can be too exploitative early on.
    # UCB is more exploratory.
    
    # A good compromise is often a weighted sum or a specific variant.
    # Let's try a "Log Expected Improvement" or "Scaled EI"?
    
    # Actually, for small sample sizes (30 evals), EI tends to work very well if the GP is well-calibrated.
    # The GP uses Matern-5/2 with length scale 0.2 on [0,1] scaled coords.
    # This means the GP is fairly wiggly (short length scale), which helps with local optima.
    
    # Let's try EI alone first.
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() - start_time > 290:
         return ei
    
    return ei
