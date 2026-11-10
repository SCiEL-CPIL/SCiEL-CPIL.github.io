import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Strategy: Use a high-resolution binomial tree (1200 steps) to compute
    # optimal exercise boundaries via backward induction.
    # Improvement: Use a much finer grid focused on the region [0, 1.2*K]
    # where the exercise boundary is located. This reduces interpolation error
    # significantly in the critical area.
    
    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise dates: steps 100, 200, 300, ..., 1200
    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}
    
    # Grid for S values
    # Focus on [0, 1.2*K] as the boundary is <= K and value is ~0 for S >> K.
    # Using 20,000 points gives ~0.006 resolution near K=100, vs 0.06 before.
    S_min = 0.0
    S_max = 1.2 * K
    n_grid = 20000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity (step n_steps)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries
    boundaries = np.zeros(n_exercises + 1)
    
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        # np.interp clamps to boundary values if out of range, which is correct
        # as V->0 for S->inf and V->K for S->0 (roughly, though tree handles it).
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        if step in exercise_steps:
            idx = exercise_steps[step]
            diff = intrinsic - continuation
            
            # Find the last index where diff >= 0
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                if last_idx < n_grid - 1:
                    diff0 = diff[last_idx]
                    diff1 = diff[last_idx + 1]
                    if diff1 != diff0:
                        t = diff0 / (diff0 - diff1)
                        boundaries[idx] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                    else:
                        boundaries[idx] = S_grid[last_idx]
                else:
                    boundaries[idx] = S_grid[-1]
            else:
                boundaries[idx] = 0.0
        
        V_next = V_current
        
        # Time check
        if step % 500 == 0:
            if time.time() > deadline - 5:
                # If running out of time, break and use current boundaries
                # Fill remaining with K (conservative for puts? No, K is max boundary)
                # Actually, if we break early, we have boundaries for later dates.
                # Earlier dates are missing. We should probably not break unless absolutely necessary.
                # 20k grid * 1200 steps is ~24M ops. Should be < 5s.
                pass

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
