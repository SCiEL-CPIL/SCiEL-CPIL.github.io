import numpy as np
import time

class BermudanPolicy:
    def __init__(self, coeffs, K, r, dt):
        self.coeffs = coeffs
        self.K = K
        self.r = r
        self.dt = dt

    def exercise(self, current_state, time_index):
        # Handle the last step
        if time_index == len(self.coeffs) - 1:
            return current_state < self.K
        
        c = self.coeffs[time_index]
        if c is None:
            return current_state < self.K
            
        S = current_state
        # Compute predicted continuation value: c[0] + c[1]*S + c[2]*S^2
        cont_val = c[0] + c[1] * S + c[2] * S * S
        
        # Immediate payoff
        payoff = np.maximum(self.K - S, 0)
        
        # Exercise if immediate payoff is greater than continuation value
        return payoff > cont_val

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Extract parameters
    K = contract['K']
    r = contract['r']
    T = contract['T']
    n_dates = contract.get('n_dates', 12)
    
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim != 2:
        return _fallback_policy(K)
    
    # Determine orientation
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T
        
    n_paths, n_steps = paths.shape
    
    # Ensure we have the right number of steps
    if n_steps > n_dates:
        paths = paths[:, -n_dates:]
        n_steps = n_dates
    elif n_steps < n_dates:
        # Pad with zeros? Or just use what we have?
        # This shouldn't happen.
        pass

    dt = T / n_dates
    
    coeffs = [None] * n_dates
    
    # Initialize V at the last step
    S_last = paths[:, -1]
    V_next = np.maximum(K - S_last, 0)
    
    # Backward induction
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_t = paths[:, t]
        
        # Discount the next value
        V_disc = np.exp(-r * dt) * V_next
        
        # Features: 1, S, S^2
        # We solve for c in [1, S, S^2] @ c = V_disc
        # Using normal equations or lstsq. lstsq is safer.
        features = np.column_stack((np.ones_like(S_t), S_t, S_t**2))
        
        try:
            coeff, _, _, _ = np.linalg.lstsq(features, V_disc, rcond=None)
            coeffs[t] = coeff
        except:
            coeffs[t] = np.array([np.mean(V_disc), 0.0, 0.0])
            
        # Predicted continuation
        # V_cont_pred = features @ coeff
        # Optimal value
        P_t = np.maximum(K - S_t, 0)
        V_cont_pred = coeff[0] + coeff[1] * S_t + coeff[2] * S_t * S_t
        V_t = np.maximum(P_t, V_cont_pred)
        
        V_next = V_t
        
    return BermudanPolicy(coeffs, K, r, dt)

def _fallback_policy(K):
    class SimplePolicy:
        def __init__(self, K):
            self.K = K
        def exercise(self, current_state, time_index):
            return current_state < self.K
    return SimplePolicy(K)
