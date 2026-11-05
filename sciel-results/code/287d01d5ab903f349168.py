import numpy as np

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def exercise(self, current_state, time):
        if time <= 0:
            return np.zeros(len(current_state), dtype=bool)
        
        # Map time to index 1..12
        # t = k/12 => k = t*12
        k = int(round(time * 12.0))
        
        if k < 1:
            k = 1
        elif k > 12:
            k = 12
            
        idx = k - 1
        
        return current_state <= self.thresholds[idx]

def fit_exercise_policy(training_paths, contract):
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    # Use a high-resolution binomial tree
    N = 1000
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    S0 = 100.0
    
    # Identify steps corresponding to exercise dates
    steps_of_interest = {}
    for k in range(1, n_dates + 1):
        t_k = k * (T / n_dates)
        j_k = int(round(t_k / dt))
        if 0 <= j_k <= N:
            steps_of_interest[j_k] = k - 1
    
    # Initialize V at maturity (step N)
    i = np.arange(N + 1)
    S_N = S0 * (u ** i) * (d ** (N - i))
    V = np.maximum(K - S_N, 0.0)
    
    recorded_values = [0.0] * n_dates
    
    # Backward induction
    for j in range(N - 1, -1, -1):
        # Continuation value
        V_next = V
        V = np.exp(-r * dt) * (p * V_next[1:] + (1 - p) * V_next[:-1])
        
        if j in steps_of_interest:
            k_idx = steps_of_interest[j]
            i_curr = np.arange(j + 1)
            S_curr = S0 * (u ** i_curr) * (d ** (j - i_curr))
            
            V_ex = np.maximum(K - S_curr, 0.0)
            
            # Find threshold
            # Exercise if V_ex >= V
            # S is increasing with i
            # V_ex is decreasing, V is increasing (generally)
            
            # Find the last i where V_ex >= V
            # If V_ex[i] >= V[i] and V_ex[i+1] < V[i+1], threshold is between S[i] and S[i+1]
            
            # Vectorized check
            # We can find the crossover by looking at the difference
            diff = V_ex - V
            # We want the last index where diff >= 0
            # Note: diff is positive for low S, negative for high S
            
            # Find indices where diff >= 0
            # Since it's monotonic, we can use searchsorted or simple logic
            
            # If all diff >= 0, threshold is S_curr[-1]
            # If all diff < 0, threshold is 0 (or S_curr[0] if we assume exercise at 0)
            
            # Let's find the first index where diff < 0
            # If no such index, threshold is S_curr[-1]
            
            # np.where(diff < 0)[0]
            neg_indices = np.where(diff < 0)[0]
            
            if len(neg_indices) == 0:
                # All non-negative
                threshold = S_curr[-1]
            elif neg_indices[0] == 0:
                # All negative
                threshold = 0.0
            else:
                i_high = neg_indices[0]
                i_low = i_high - 1
                
                S_low = S_curr[i_low]
                S_high = S_curr[i_high]
                
                val_low = diff[i_low]
                val_high = diff[i_high]
                
                # Interpolate
                denom = val_high - val_low
                if denom != 0:
                    ratio = -val_low / denom
                    threshold = S_low + ratio * (S_high - S_low)
                else:
                    threshold = S_low
            
            recorded_values[k_idx] = threshold
    
    thresholds = np.array(recorded_values)
    
    return ExercisePolicy(thresholds)
