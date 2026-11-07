import numpy as np
import time
from scipy.stats import norm

# Global state
_acq_state = {
    'start_time': None,
    'call_count': 0
}

def acquisition(mu, sigma, f_best, cost):
    # Initialize state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    # Increment call count
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:
        # Fallback: simple Cost-Aware EI
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        z = (f_best - mu) / s
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        ei = (f_best - mu) * Phi_z + s * phi_z
        ei = np.maximum(ei, 0.0)
        score = ei / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Calculate EI
    z = (f_best - mu) / s
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + s * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Calculate budget-scaled uncertainty bonus
    # Estimate max calls: Budget 30, Initial cost ~7.5, Avg cost ~1.5 -> ~15 calls
    max_calls = 15.0
    k = _acq_state['call_count']
    
    # Decay factor: 1.0 at start, 0.0 at end
    decay = max(0.0, 1.0 - k / max_calls)
    
    # Base beta for uncertainty bonus
    # Tuned to be significant early on
    beta_0 = 1.0
    beta = beta_0 * decay
    
    # Score = (EI + beta * sigma) / cost
    # This encourages exploration (high sigma) in cheap regions early on
    score = (ei + beta * s) / c
    
    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
