"""Bermudan exercise policy using binomial tree backward induction."""
import numpy as np
import time

# EVOLVE-BLOCK-START
class ExercisePolicy:
    def __init__(self, critical_prices, exercise_times):
        self.critical_prices = critical_prices
        self.exercise_times = exercise_times

    def exercise(self, current_state, time):
        # current_state: array of spot prices
        # time: scalar time value
        # Find the closest exercise time index
        # exercise_times are sorted ascending
        
        # Handle scalar or array time
        if np.isscalar(time):
            idx = np.argmin(np.abs(self.exercise_times - time))
            return current_state < self.critical_prices[idx]
        else:
            # If time is an array, vectorize
            # This case is less likely but handled for robustness
            # Assuming current_state and time are aligned
            idxs = np.array([np.argmin(np.abs(self.exercise_times - t)) for t in time])
            return current_state < self.critical_prices[idxs]

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Binomial tree parameters
    # Use 1200 steps to match reference diagnostic resolution
    N = 1200
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Exercise dates in terms of tree steps
    # 12 exercise dates, so every 100 steps
    steps_per_exercise = N // n_exercise
    exercise_steps = np.arange(1, n_exercise + 1) * steps_per_exercise
    exercise_steps_set = set(exercise_steps)
    
    # Initialize value array
    V = np.zeros((N + 1, N + 1))
    
    # Maturity values
    j_range = np.arange(N + 1)
    # S_N_j = S0 * u^j * d^(N-j)
    # We use S0=100 for absolute price scaling
    S0 = 100.0
    S_N = S0 * (u ** j_range) * (d ** (N - j_range))
    V[N, :] = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(n_exercise)
    exercise_time_vals = np.zeros(n_exercise)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        # Compute continuation value
        # V[i, 0..N-i] depends on V[i+1, 0..N-i+1]
        V[i, 0:N-i+1] = disc * (p * V[i+1, 1:N-i+2] + (1-p) * V[i+1, 0:N-i+1])
        
        # Check if this is an exercise date
        if i in exercise_steps_set:
            idx = (i // steps_per_exercise) - 1
            exercise_time_vals[idx] = i * dt
            
            # Compute spot prices at this time step
            j_range_i = np.arange(N - i + 1)
            S_i = S0 * (u ** j_range_i) * (d ** (i - j_range_i))
            intrinsic = np.maximum(K - S_i, 0.0)
            
            # Determine exercise boundary
            # Exercise if intrinsic >= continuation (V[i, :] before update)
            exercise_mask = intrinsic >= V[i, 0:N-i+1]
            
            if np.any(exercise_mask):
                true_indices = np.where(exercise_mask)[0]
                if len(true_indices) > 0:
                    j_max = true_indices[-1]
                    critical_prices[idx] = S_i[j_max]
                else:
                    critical_prices[idx] = 0.0
            else:
                critical_prices[idx] = 0.0
            
            # Update value with max(intrinsic, continuation)
            V[i, 0:N-i+1] = np.maximum(intrinsic, V[i, 0:N-i+1])
            
        # Time check
        if time.time() > deadline - 5.0:
            # If we run out of time, we break. 
            # The critical prices for earlier dates (lower i) will be 0.
            # This is a failure mode, but 1200 steps is fast.
            break
            
    return ExercisePolicy(critical_prices, exercise_time_vals)
# EVOLVE-BLOCK-END
