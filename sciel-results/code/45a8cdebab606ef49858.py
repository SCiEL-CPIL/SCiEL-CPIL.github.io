import numpy as np
import time

# Global state for tracking the optimization step within a seed
_state = {
    "call_count": 0,
    "prev_f_best": None,
    "start_time": None,
    "total_bo_steps": 25  # 30 total evaluations - 5 initial = 25 BO steps
}

# Cached values for efficient computation
_cache = {
    "mu_min": None,
    "mu_max": None,
    "sigma_max": None,
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses an adaptive Upper Confidence Bound (UCB) strategy with linear decay.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Std dev of the GP posterior, shape (N,)
    f_best (float): Best observed value so far (minimum).
    
    Returns:
    np.ndarray: Scores for each candidate, higher is better.
    """
    global _state
    
    # Initialize start time if not set
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    # Check deadline guard: if we are close to the limit, return a safe fallback
    # The limit is 330s, we use a margin of 5s.
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        # Fallback: return pure exploitation scores to ensure validity and speed
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Detect if this is a new seed (new optimization run)
    # f_best is non-increasing within a single seed. 
    # If it increases significantly, or if we've exceeded the max steps, it's a new seed.
    is_new_seed = False
    if _state["prev_f_best"] is not None:
        if f_best > _state["prev_f_best"] + 1e-9 or _state["call_count"] >= _state["total_bo_steps"]:
            is_new_seed = True
            
    if is_new_seed:
        _state["call_count"] = 0
        _cache["mu_min"] = None
        _cache["mu_max"] = None
        _cache["sigma_max"] = None
    else:
        _state["call_count"] += 1
        
    _state["prev_f_best"] = f_best
    
    # Ensure step count is within valid range [0, 24]
    current_step = min(_state["call_count"], _state["total_bo_steps"] - 1)
    
    # Calculate progress in the optimization loop [0, 1]
    # Step 0 is the first BO step (after initial 5 points)
    progress = current_step / max(_state["total_bo_steps"] - 1, 1)
    
    # Adaptive exploration rate beta for UCB
    # Use a smoother decay curve that maintains more exploration longer
    # For Hartmann 6D, we want to explore more broadly in early/mid steps
    beta_start = 2.5
    beta_end = 0.2
    # Use quadratic decay for smoother transition
    beta = beta_start + (beta_end - beta_start) * progress ** 1.5
    
    # UCB Score for minimization:
    # We want to minimize f(x). The GP posterior gives mu and sigma.
    # The Lower Confidence Bound (LCB) for minimization is: LCB(x) = mu(x) - beta * sigma(x)
    # We want to MINIMIZE LCB, which is equivalent to MAXIMIZING (beta * sigma(x) - mu(x))
    # So, score(x) = beta * sigma(x) - mu(x)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma or mu
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute base UCB score
    scores = beta * sigma - mu
    
    # Add a small diversity bonus in early steps to encourage exploration
    # This helps escape local optima by slightly boosting candidates with higher uncertainty
    if current_step < 8:  # First 8 BO steps
        # Normalize sigma to [0, 1] for the diversity bonus
        sigma_min = np.min(sigma)
        sigma_max = np.max(sigma)
        if sigma_max > sigma_min + 1e-12:
            sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = np.zeros_like(sigma)
        
        # Diversity bonus: small weight that decays over the first 8 steps
        diversity_weight = 0.15 * (1.0 - current_step / 8.0)
        scores = scores + diversity_weight * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
