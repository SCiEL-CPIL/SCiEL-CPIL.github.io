import numpy as np
import time

# Global state for tracking the optimization step within a seed
_state = {
    "call_count": 0,
    "prev_f_best": None,
    "start_time": None,
    "total_bo_steps": 25  # 30 total evaluations - 5 initial = 25 BO steps
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a pure Upper Confidence Bound (UCB) strategy with high exploration.
    
    For minimization, the UCB score is: score(x) = -mu(x) - kappa * sigma(x)
    where kappa is a high exploration constant to aggressively target regions
    of high posterior variance, maximizing the probability of discovering the
    global minimum in the highly multimodal Hartmann 6D landscape.
    
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
    else:
        _state["call_count"] += 1
        
    _state["prev_f_best"] = f_best
    
    # Ensure step count is within valid range [0, 24]
    current_step = min(_state["call_count"], _state["total_bo_steps"] - 1)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma or mu
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Pure UCB strategy with high exploration constant
    # score(x) = -mu(x) - kappa * sigma(x)
    # Higher kappa means more aggressive exploration of high-variance regions
    kappa = 5.0
    scores = -mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
