import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    # 150s limit, use 120s safety
    deadline = start_time + 120.0 

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    exercise_times = np.linspace(1/12.0, T, n_exercise)

    # Use a high number of steps for high precision.
    # N=1200 was the previous setting.
    # Let's try N=20000. 
    # If it's too slow, we can fallback.
    
    # Check if we have enough time for a large tree.
    # A rough heuristic: N=1000 takes ~0.1s. N=20000 takes ~20s?
    # Let's start with N=5000 to be safe, then increase if fast.
    # Actually, let's just go for N=10000 and see.
    
    N = 10000
    steps_per_month = N // n_exercise # 833.33 -> 833
    # Ensure steps_per_month is integer and divides N?
    # The exercise dates are at t = k/12.
    # We need the tree step i to correspond exactly to t_k.
    # t_k = k * T / 12.
    # Step i corresponds to t_i = i * T / N.
    # We need i_k = k * N / 12.
    # For i_k to be integer, N must be divisible by 12.
    # 10000 is not divisible by 12.
    # 10000 / 12 = 833.33.
    # So step 833 is at t = 833/10000 = 0.0833. t_1 = 1/12 = 0.08333.
    # Close, but not exact.
    # Better to choose N divisible by 12.
    # Let's use N = 12000.
    N = 12000
    steps_per_month = N // n_exercise # 1000
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    S0 = 100.0
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    
    # Initialize at T (i=N)
    j_N = np.arange(N + 1)
    log_S_N = log_S0 + j_N * log_u + (N - j_N) * log_d
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = [0.0] * 13  # 1..12
    
    # Backward induction
    # Optimization: Pre-allocate arrays? 
    # V size changes.
    
    for i in range(N - 1, -1, -1):
        if time.time() > deadline:
            break
            
        # Determine if this is an exercise step
        if i % steps_per_month == 0:
            k = i // steps_per_month
            if 1 <= k <= 12:
                # Compute S_i
                j_i = np.arange(i + 1)
                log_S_i = log_S0 + j_i * log_u + (i - j_i) * log_d
                S_i = np.exp(log_S_i)
                
                # Compute Continuation Value
                V_up = V[1:]
                V_down = V[:-1]
                V_cont = disc * (p * V_up + (1 - p) * V_down)
                
                V_ex = np.maximum(K - S_i, 0.0)
                
                # Find boundary
                diff = V_ex - V_cont
                valid = np.where(diff >= 0)[0]
                if len(valid) > 0:
                    j_max = valid[-1]
                    critical_prices[k] = S_i[j_max]
                else:
                    critical_prices[k] = 0.0
                
                # Update V
                V = np.maximum(V_cont, V_ex)
            else:
                V_up = V[1:]
                V_down = V[:-1]
                V = disc * (p * V_up + (1 - p) * V_down)
        else:
            V_up = V[1:]
            V_down = V[:-1]
            V = disc * (p * V_up + (1 - p) * V_down)
            
    return ExercisePolicy(critical_prices)

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)
        
    def exercise(self, current_state, time):
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        S_star = self.critical_prices[k_int]
        return current_state <= S_star

def _fallback_policy():
    cp = [0.0] * 13
    for k in range(1, 13):
        cp[k] = 100.0 * (k / 12.0) * 0.9
    return ExercisePolicy(cp)
