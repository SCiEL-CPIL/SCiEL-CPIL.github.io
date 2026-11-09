import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses a hybrid strategy that adapts to the current state of the search:
    1. If the best value found is still high (early stage), use Expected Improvement (EI)
       to balance exploration and exploitation effectively.
    2. If the best value found is low (late stage, near convergence), switch to 
       Posterior Mean Minimization (PMM) to aggressively exploit the GP's predictive mean,
       which helps fine-tune the location of the minimum when uncertainty is low.
    
    This structural change addresses potential stagnation by ensuring the search
    doesn't get stuck in broad exploration when it should be refining the solution.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score for EI
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Clip Z to avoid numerical issues
    Z_clipped = np.clip(Z, -10, 10)
    
    phi_Z = norm.pdf(Z_clipped)
    Phi_Z = norm.cdf(Z_clipped)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
         return EI

    # Structural Change: Adaptive Strategy
    # If we are very close to the minimum (f_best is small), pure EI might be too conservative
    # or suffer from numerical issues if sigma is tiny.
    # In 1D sphere minimization, once we are close, the posterior mean is a very good guide.
    # Let's use a threshold to decide when to switch to PMM (Posterior Mean Minimization).
    
    # Threshold: if f_best is less than 1.0, we are likely in the exploitation phase.
    # The sphere function is x^2. So f_best < 1.0 means |x| < 1.0.
    # At this point, aggressive exploitation of the mean is often better.
    
    threshold = 1.0
    
    if f_best < threshold:
        # Use PMM: minimize the posterior mean
        # Score is -mu (higher score = lower mu)
        scores = -mu
    else:
        # Use EI
        scores = EI
        
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
