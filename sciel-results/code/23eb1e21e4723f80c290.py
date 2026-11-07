"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for safety margin
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid approach combining Expected Improvement (EI) and
    a Lower Confidence Bound (LCB) term to balance exploitation and exploration.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Safety check: if we are getting close to the 300s limit (330s hard kill),
    # we might want to fail fast or return a simple result. 
    # However, the calculation here is vectorized and fast for 2000 points.
    # The main cost is in the GP posterior computation which happens outside this function.
    # This function itself should be negligible in time.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # Standard EI formula for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Gaussian Process Lower Confidence Bound (GP-LCB)
    # We want to minimize, so we look at the lower bound of the distribution.
    # LCB = mu - beta * sigma
    # We want to maximize the acquisition score, so we negate it or use a different form.
    # Commonly, for minimization, we maximize: -(mu - beta * sigma) = beta * sigma - mu
    # Or simply use the LCB value itself if the framework expects "higher is better" for the score
    # and the score is used to pick the point with the *lowest* predicted value?
    # The task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So if we use LCB = mu - beta*sigma, a *lower* LCB is better.
    # Therefore, the acquisition score for LCB should be -LCB = beta*sigma - mu.
    
    # Choose beta. A common heuristic is beta = 2 or 3. 
    # In high dimensions, more exploration is often needed.
    # Let's try a fixed beta first, or an adaptive one if we had more context.
    # Since we don't have the number of evaluations passed directly, we can't easily adapt beta 
    # based on t unless we infer it. But we can use a static value.
    # Let's use beta = 2.5.
    beta = 2.5
    lcb_score = beta * sigma_safe - mu
    
    # 3. Hybrid
    # We need to combine these. EI and LCB have different scales.
    # EI is in objective units. LCB is in objective units.
    # However, EI tends to be small near the optimum, while LCB can be large in unexplored areas.
    # A simple weighted sum might work.
    # Let's try a 50/50 mix or tune the weights.
    # w_ei * EI + w_lcb * LCB_score
    
    # To make them comparable, we might need to scale them, but standard practice often
    # just sums them or uses one primarily.
    # Let's try: scores = 0.5 * ei + 0.5 * lcb_score
    
    # Actually, EI is already a measure of "how much better this point could be".
    # LCB is a measure of "how low this point could be".
    # Mixing them directly is common.
    
    # Combine with weights that favor exploration slightly more than the current program
    # to address potential stagnation in local optima, inspired by the successful 
    # exploration-focused approach in inspiration 1 but structured as a proper LCB hybrid.
    scores = 0.4 * ei + 0.6 * lcb_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
