import numpy as np
from scipy.stats import norm
import time

# We use a global flag or just check time.
# The directive says: "track elapsed time and return the best result found with a safety margin".
# Since we don't have a "best result found" state inside this function (it's stateless),
# "best result found" likely means the best score we can compute quickly.
# If we run out of time, we should return something valid (finite) quickly.
# A safe fallback is to return scores based only on cost (prefer cheap points) or random.
# But since the function is fast, we just do the math.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI) and Uncertainty (UCB-like).
    
    Args:
        mu: Mean of the GP (shape 2000,)
        sigma: Standard deviation of the GP (shape 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluation for each point (shape 2000,)
        
    Returns:
        scores: Acquisition values (shape 2000,)
    """
    start_time = time.time()
    
    # Safety margin: if we've already used > 290s of the 300s budget (hypothetically), 
    # we should return immediately. 
    # In reality, this function is called many times, so we can't track total time here 
    # unless we use a global variable. 
    # However, the directive implies we should be cautious.
    # Let's assume the "deadline" refers to the total run. 
    # Since we can't know the total elapsed time without a global state, 
    # and the function is O(N) with N=2000, it's fast.
    
    try:
        # Ensure inputs are floats
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Avoid division by zero
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # Calculate z-score
        z = (f_best - mu) / sigma_safe
        
        # Expected Improvement (EI)
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # Note: For minimization, improvement is f_best - f(x).
        # If f(x) < f_best, improvement is positive.
        # The formula above is correct for minimization EI.
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        
        # UCB-like exploration term
        # We want to explore high uncertainty.
        # Let's add a term proportional to sigma.
        # Beta controls the strength.
        beta = 2.0
        ucb_term = beta * sigma_safe
        
        # Combine
        # We weight EI and UCB. 
        # EI is already a measure of expected gain.
        # Adding raw sigma might dominate if sigma is large.
        # Let's normalize or just sum.
        # The inspiration said "Blending EI and UCB".
        # Let's try: score = (ei + ucb_term) / cost
        
        raw_score = (ei + ucb_term)
        
        # Normalize by cost
        # Cost is in [1, 2].
        scores = raw_score / cost
        
        # Ensure finite values
        # If any NaN or Inf, replace with a very low score (or 0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 0.1: # If this took > 100ms, something is wrong
             # Fallback to simple cost-based score
             scores = 1.0 / cost
             scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
             
        return scores

    except Exception:
        # In case of any error, return a valid array
        # Prefer cheap points
        try:
             cost_arr = np.asarray(cost, dtype=np.float64)
             return 1.0 / np.maximum(cost_arr, 1e-9)
        except:
             return np.zeros(2000)
