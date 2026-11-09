import numpy as np

# Module-level state
_state = {
    'count': 0,
    'last_f_best': None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Decaying Lower Confidence Bound (LCB).
    """
    # Update state
    if _state['last_f_best'] is not None:
        # If f_best has increased significantly, assume a new run started
        # or the previous run was bad and this one started with better random points?
        # In minimization, f_best should decrease. An increase means new run.
        if f_best > _state['last_f_best']:
            _state['count'] = 0
        
        # Safety: if count exceeds max expected steps (25) + buffer, reset
        if _state['count'] > 30:
            _state['count'] = 0
    else:
        _state['count'] = 0
        
    _state['count'] += 1
    _state['last_f_best'] = f_best
    
    # Decay schedule
    # Total loop steps = 25
    max_steps = 25
    # Clamp count to 1..max_steps
    t = min(_state['count'], max_steps)
    
    # Linear decay from beta_start to beta_end
    beta_start = 2.5
    beta_end = 0.5
    
    progress = t / max_steps
    beta = beta_start * (1.0 - progress) + beta_end * progress
    
    # LCB score: -mu + beta * sigma
    # We want to maximize this score.
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
