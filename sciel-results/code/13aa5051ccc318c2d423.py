import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        # critical_prices is a list or array of length 13 (index 0 unused or 1..12)
        # Let's assume 1-indexed for time steps 1..12
        self.critical_prices = np.array(critical_prices)
        
    def exercise(self, current_state, time):
        # current_state: array-like of asset prices
        # time: float
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        # Check bounds
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
        
        # Get critical price for this time step
        # critical_prices is indexed 1..12
        cp = self.critical_prices[k_int]
        
        # Exercise if S < Critical Price
        return np.array(current_state) < cp

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # 1. Parse training paths
    # training_paths is likely a list of lists or a 2D array
    # Shape: (N, 13) corresponding to t=0..12
    S = np.array(training_paths)
    if S.ndim == 1:
        S = S.reshape(-1, 13)
        
    N = S.shape[0]
    # Ensure we have enough paths
    if N < 100:
        # Fallback to simple heuristic if data is insufficient
        # Put: Exercise if S < K
        cp = np.full(13, 100.0)
        return ExercisePolicy(cp)
        
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    dt = T / n_steps
    disc = np.exp(-r * dt)
    
    # V[i, k] is the value at step k for path i
    V = np.zeros((N, n_steps + 1))
    # Exercise mask: True if we exercise at step k
    ex_mask = np.zeros((N, n_steps + 1), dtype=bool)
    
    # Step 12 (t=1.0): Always exercise if in the money
    V[:, 12] = np.maximum(K - S[:, 12], 0.0)
    ex_mask[:, 12] = (S[:, 12] < K)
    
    # Backward Induction
    # We estimate continuation value E[V_{k+1} | S_k] using local averaging on training paths
    # Number of neighbors for local average
    n_neighbors = 50
    
    for k in range(11, -1, -1):
        # Check time
        if time.time() > deadline:
            break
            
        # Sort indices by S[:, k]
        sort_idx = np.argsort(S[:, k])
        S_sorted = S[sort_idx, k]
        V_next_sorted = V[sort_idx, k + 1]
        
        # Compute local average of V_next for each point
        # We can use a sliding window or simple nearest neighbors
        # For simplicity and speed, let's use a fixed window around each point in sorted order
        # This is an approximation of E[V_next | S_k]
        
        C = np.zeros(N)
        
        # Efficient local averaging
        # For each i in 0..N-1, average V_next_sorted[max(0, i-n//2) : min(N, i+n//2+1)]
        # This is O(N * n_neighbors)
        
        half_win = n_neighbors // 2
        for i in range(N):
            start = max(0, i - half_win)
            end = min(N, i + half_win + 1)
            C[sort_idx[i]] = np.mean(V_next_sorted[start:end])
            
        # Discounted continuation value
        C_disc = disc * C
        
        # Value at step k
        intrinsic = np.maximum(K - S[:, k], 0.0)
        V[:, k] = np.maximum(intrinsic, C_disc)
        
        # Exercise decision
        ex_mask[:, k] = (intrinsic >= C_disc)
        
    # Determine Critical Prices
    # For each k, B_k is the max S where exercise is optimal
    critical_prices = np.zeros(13)
    
    for k in range(1, 13):
        # Get S and ex_mask for this k
        s_k = S[:, k]
        ex_k = ex_mask[:, k]
        
        if np.any(ex_k):
            # The boundary is the maximum S that exercises
            # In practice, there might be noise, so taking the max is a conservative estimate of the boundary
            # Or we could take a percentile, e.g., 95th percentile of exercising prices
            # Max is the safest bet for "exercise if below"
            critical_prices[k] = np.max(s_k[ex_k])
        else:
            critical_prices[k] = 0.0
            
    # Ensure monotonicity? The critical price for a put should increase as time decreases (further from expiry, higher strike is justified? No, for a put, the critical price usually increases as we move away from expiry? 
    # Actually, for a European put, no exercise. For Bermudan, as t -> 0, the option is more valuable, so we might exercise at higher prices? 
    # Wait, if I have a put with K=100, and S=90. If I have a long time to expiry, I might wait. If I have short time, I might exercise.
    # So B_k should generally decrease as k increases (as we approach expiry).
    # Let's enforce monotonicity: B_k >= B_{k+1}
    for k in range(11, 0, -1):
        if critical_prices[k] < critical_prices[k+1]:
            critical_prices[k] = critical_prices[k+1]
            
    return ExercisePolicy(critical_prices)
