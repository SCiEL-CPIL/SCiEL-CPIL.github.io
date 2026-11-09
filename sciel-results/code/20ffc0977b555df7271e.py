"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track call count and detect new seeds
_state = {
    'n_calls': 0,
    'prev_f_best': None,
    'start_time': time.time()
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    This is a structural change from Expected Improvement (EI). LCB explicitly
    controls the exploration-exploitation trade-off via a parameter beta.
    We use an adaptive beta that starts high (exploratory) and decays to low
    (exploitative) as the number of evaluations increases. This helps escape
    local minima in the early stages of the optimization of the multi-modal
    Styblinski-Tang function.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means
    - sigma: array of shape (2000,) with posterior standard deviations
    - f_best: scalar, the best observed value so far
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Convert to numpy arrays for efficient vectorized operations
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Detect if a new seed/run has started
    # f_best typically decreases within a run. If it increases significantly, it's likely a new run.
    if _state['prev_f_best'] is not None and f_best > _state['prev_f_best'] + 1e-6:
        _state['n_calls'] = 0
        _state['start_time'] = time.time()
        
    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1
    
    # Deadline Guard
    # The total budget is 330s. If we are close to the limit, return quickly.
    # Pure exploitation (-mu) is the fastest to compute and safe.
    elapsed = time.time() - _state['start_time']
    if elapsed > 320.0:
        score = -mu
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Adaptive Beta for LCB
    # Start with high exploration (beta=5.0) and decay to exploitation (beta=0.5)
    # Using an exponential decay with time constant tau=10 evaluations.
    beta_max = 5.0
    beta_min = 0.5
    tau = 10.0
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-n / tau)
    
    # LCB for minimization: a(x) = mu(x) - beta * sigma(x)
    # We want to maximize the acquisition score, so we return:
    # score(x) = -a(x) = beta * sigma(x) - mu(x)
    
    score = beta * sigma - mu
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
