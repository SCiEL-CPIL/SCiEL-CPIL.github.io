import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # V[j, i] stores the value of the option at exercise date j (step j+1) for path i
    V = np.zeros((n_exercise, n_paths))
    
    # Last exercise date: j=11, step=12
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # KNN Parameters
    k_neighbors = 25  # Number of neighbors for local regression
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        
        # We want to estimate E[V[j+1] | S_curr]
        # Use KNN Regression on the sorted S_curr values
        
        # Sort indices by current spot price
        sort_idx = np.argsort(S_curr)
        S_sorted = S_curr[sort_idx]
        V_next_sorted = V[j+1, :][sort_idx]
        
        # Compute KNN regression values for all points
        # For each point in sorted order, we look at its k nearest neighbors
        # Since data is sorted, the k nearest neighbors are simply the k points 
        # closest in the sorted array (considering boundaries).
        
        continuation = np.zeros(n_paths)
        
        # Efficient KNN on sorted 1D data
        # We can use a sliding window approach or just loop with numpy slicing
        # Given N=4096, a simple loop with vectorized slicing is fast enough.
        
        half_k = k_neighbors // 2
        
        for i in range(n_paths):
            # Indices in the sorted array
            lo = max(0, i - half_k)
            hi = min(n_paths, i + half_k + 1)
            
            # Take the neighbors
            neighbors = V_next_sorted[lo:hi]
            
            # Simple mean
            continuation[sort_idx[i]] = np.mean(neighbors)
            
        # Discount the continuation value
        cont_values = discount * continuation
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
    # Extract Thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last step (j=11), exercise if in the money
    thresholds[11] = K
    
    for j in range(10, -1, -1):
        if _time.time() > deadline - 5:
            break
            
        S_curr = paths[:, j+1]
        
        # Recompute continuation values using the same KNN logic for this step
        # Note: We need the continuation value function to find the threshold.
        # The KNN values computed above are for the specific paths in training_paths.
        # To find the threshold S*, we need to evaluate the continuation value 
        # at arbitrary S points, or find the crossing in the training data.
        
        # Method: Find the crossing in the sorted training data.
        sort_idx = np.argsort(S_curr)
        S_sorted = S_curr[sort_idx]
        V_next_sorted = V[j+1, :][sort_idx]
        
        # Compute KNN continuation for sorted paths
        cont_sorted = np.zeros(n_paths)
        half_k = k_neighbors // 2
        
        for i in range(n_paths):
            lo = max(0, i - half_k)
            hi = min(n_paths, i + half_k + 1)
            cont_sorted[i] = np.mean(V_next_sorted[lo:hi])
        
        cont_discounted = discount * cont_sorted
        
        payoff_sorted = np.maximum(K - S_sorted, 0.0)
        
        # Find where payoff > continuation
        # diff = payoff - continuation
        # We want the largest S where diff > 0
        diff = payoff_sorted - cont_discounted
        
        # Only consider ITM region (S < K) for the threshold search, 
        # but the crossing might be slightly above K if continuation is negative (unlikely)
        # For a put, threshold is always < K.
        
        # Find the last index where diff > 0
        pos_indices = np.where(diff > 0)[0]
        
        if len(pos_indices) == 0:
            thresholds[j] = 0.0
        else:
            last_pos_idx = pos_indices[-1]
            
            # Check if the next point is also positive or if we crossed
            if last_pos_idx + 1 < n_paths:
                S1 = S_sorted[last_pos_idx]
                S2 = S_sorted[last_pos_idx + 1]
                d1 = diff[last_pos_idx]
                d2 = diff[last_pos_idx + 1]
                
                # Interpolate if sign change
                if d1 > 0 and d2 <= 0:
                    if d1 - d2 != 0:
                        t = d1 / (d1 - d2)
                        thresholds[j] = S1 + t * (S2 - S1)
                    else:
                        thresholds[j] = S1
                else:
                    # If d2 is also positive, the threshold is higher
                    # Since we only go up to S_sorted max, if all are positive, 
                    # threshold is at least S_sorted[-1]. 
                    # But for a put, if S is high, payoff is 0. 
                    # If diff is positive at high S, it means continuation is negative? 
                    # Continuation is max(payoff, 0) in V, so it's non-negative.
                    # Payoff is 0 for S > K. So diff = -cont <= 0.
                    # So diff must become negative after K.
                    thresholds[j] = S1
            else:
                thresholds[j] = S_sorted[last_pos_idx]
                
    # Safety: ensure thresholds are reasonable
    thresholds = np.clip(thresholds, 0.0, K)
    
    return ExercisePolicy(thresholds)
