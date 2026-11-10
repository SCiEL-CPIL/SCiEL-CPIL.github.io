import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    S0 = contract.get('S0', 100.0)

    # Use 5000 steps for high accuracy. 
    # The inner loop is O(N) with very small constant (array shifts/adds).
    n_steps = 5000
    dt = T / n_steps
    delta = sigma * math.sqrt(dt)
    
    df = math.exp(-r * dt)
    u = math.exp(delta)
    d = math.exp(-delta)
    p = (math.exp(r * dt) - d) / (u - d)

    exercise_steps_map = {}
    for i in range(1, n_exercises + 1):
        step = i * n_steps // n_exercises
        exercise_steps_map[step] = i

    # Grid in Log-Space
    buffer = 10 * delta
    Y_min = math.log(S0) - n_steps * delta - buffer
    Y_max = math.log(S0) + n_steps * delta + buffer
    
    n_grid = int((Y_max - Y_min) / delta) + 1
    Y_grid = Y_min + np.arange(n_grid) * delta
    S_grid = np.exp(Y_grid)
    
    V_next = np.maximum(K - S_grid, 0.0)
    V_curr = np.empty_like(V_next)
    
    boundaries = np.zeros(n_exercises + 1)
    
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation
        V_curr[1:-1] = df * (p * V_next[2:] + (1.0 - p) * V_next[:-2])
        V_curr[0] = df * p * V_next[1]
        V_curr[-1] = df * (1.0 - p) * V_next[-2]
        
        V_curr = np.maximum(V_curr, intrinsic)
        
        if step in exercise_steps_map:
            idx = exercise_steps_map[step]
            # Find boundary
            diff = intrinsic - V_curr
            # Exercise when intrinsic > continuation, i.e., diff > 0
            # For a put, this happens at low S (low Y, low index)
            # So we look for the transition from diff > 0 to diff < 0
            # As index increases, S increases, Intrinsic decreases, Continuation increases.
            # So diff goes from positive to negative.
            
            # Find the last index where diff >= 0
            # Since diff is monotonic decreasing (mostly), we can use searchsorted or linear scan
            # But linear scan on 10k elements is fast.
            # Or use np.where
            mask = diff >= 0
            if np.any(mask):
                idxs = np.where(mask)[0]
                last_idx = idxs[-1]
                if last_idx < n_grid - 1:
                    # Interpolate between last_idx and last_idx+1
                    d0 = diff[last_idx]
                    d1 = diff[last_idx + 1]
                    if d1 != d0:
                        t = d0 / (d0 - d1)
                        Y_boundary = Y_grid[last_idx] + t * (Y_grid[last_idx + 1] - Y_grid[last_idx])
                        boundaries[idx] = math.exp(Y_boundary)
                    else:
                        boundaries[idx] = S_grid[last_idx]
                else:
                    boundaries[idx] = S_grid[-1]
            else:
                boundaries[idx] = 0.0
                
        V_next = V_curr
        
        if time.time() > deadline - 5:
            break

    boundaries[12] = K

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(boundaries, K)
