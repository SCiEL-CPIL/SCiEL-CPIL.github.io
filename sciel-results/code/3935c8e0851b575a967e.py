import numpy as np

class ExercisePolicy:
    def __init__(self, coeffs, K, r, dt):
        self.coeffs = coeffs # List of coefficient arrays for steps 1..11
        self.K = K
        self.r = r
        self.dt = dt
        self.df = np.exp(-r * dt)
        
    def exercise(self, current_state, time):
        # current_state: array of spots
        # time: time value or step index
        # Map time to step index k (1..12)
        if isinstance(time, (int, np.integer)):
            k = int(time)
        else:
            # Assume time is in years, step size 1/12
            k = int(round(time * 12))
            
        # Clamp k to valid range 1..12
        k = max(1, min(12, k))
        
        # Immediate payoff
        P = np.maximum(self.K - np.asarray(current_state), 0.0)
        
        if k == 12:
            # Last exercise date: exercise if payoff > 0
            return P > 0.0
        
        # For k < 12, use regression coefficients
        # coeffs list is indexed 0..10 corresponding to steps 1..11
        idx = k - 1
        c = self.coeffs[idx]
        
        # Calculate continuation value
        # polyval expects coefficients in descending order
        C = np.polyval(c, np.asarray(current_state)) * self.df
        C = np.maximum(C, 0.0) # Continuation value cannot be negative
        
        return P >= C

def fit_exercise_policy(training_paths, contract):
    # Parse contract parameters
    # contract likely contains K, r, sigma, T, etc.
    # Based on task description: K=100, r=0.05, sigma=0.2, T=1
    # We assume standard values if not explicitly in contract dict
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    
    # training_paths shape: (N, 13) or (N, 12)
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    N, D = paths.shape
    
    # Determine if t=0 is included
    # If D == 13, indices 0..12 are t0..t12. Exercise at 1..12.
    # If D == 12, indices 0..11 are t1..t12. Exercise at 1..12 (mapped to 0..11 in array).
    if D == 13:
        # S_k is at index k
        S = paths[:, 1:13] # Shape (N, 12)
    elif D == 12:
        # S_k is at index k-1
        S = paths[:, 0:12] # Shape (N, 12)
    else:
        # Fallback: assume last 12 columns are exercise dates
        S = paths[:, -12:]
        
    # Calculate immediate payoffs for all steps
    # P shape (N, 12)
    P = np.maximum(K - S, 0.0)
    
    df = np.exp(-r * dt)
    
    # Backward induction
    # V_next is the value at step k+1
    # We start from the last step (index 11 in S/P arrays, corresponding to step 12)
    V_next = P[:, 11].copy()
    
    coeffs = []
    
    # Loop from step 11 down to 1
    # Step k corresponds to index k-1 in S and P
    for k in range(11, 0, -1):
        idx_curr = k - 1 # Index in S/P for current step k
        idx_next = k     # Index in S/P for next step k+1 (already in V_next)
        
        S_curr = S[:, idx_curr]
        
        # Regress V_next on S_curr
        # Use degree 3 polynomial
        # Filter out any NaNs/Infs just in case
        mask = np.isfinite(S_curr) & np.isfinite(V_next)
        if np.sum(mask) < 10:
            # Fallback to constant if data is bad
            c = [np.mean(V_next)]
        else:
            c = np.polyfit(S_curr[mask], V_next[mask], 3)
            
        coeffs.append(c)
        
        # Calculate continuation value at current step
        C_curr = np.polyval(c, S_curr) * df
        C_curr = np.maximum(C_curr, 0.0)
        
        # Value at current step
        P_curr = P[:, idx_curr]
        V_curr = np.maximum(P_curr, C_curr)
        
        # Update V_next for the next iteration (which is the previous step)
        V_next = V_curr
        
    # Reverse coeffs because we built them from step 11 down to 1
    # coeffs[0] is for step 11, coeffs[1] for step 10, ..., coeffs[10] for step 1
    # We need coeffs for step 1 at index 0, step 2 at index 1, etc.
    coeffs = coeffs[::-1]
    
    return ExercisePolicy(coeffs, K, r, dt)
