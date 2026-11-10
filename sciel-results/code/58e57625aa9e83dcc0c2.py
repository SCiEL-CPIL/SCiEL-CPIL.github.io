import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 140.0  # 140s safety margin before 150s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Use a high-resolution binomial tree to compute optimal exercise boundaries
    # via backward induction. The tree has 1200 steps, with exercise at every
    # 100th step (steps 100, 200, ..., 1200).
    
    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise dates: steps 100, 200, ..., 1200
    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}
    
    # Grid for S values
    # The exercise boundary for a Bermudan put is less than K. We focus on
    # [0, 1.5*K] to have plenty of headroom, with 40,000 points for high
    # resolution near the boundary.
    S_min = 0.0
    S_max = 1.5 * K
    n_grid = 40000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity (step n_steps)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries array: boundaries[i] is the exercise boundary at exercise date i
    boundaries = np.zeros(n_exercises + 1)
    
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        if step in exercise_steps:
            idx = exercise_steps[step]
            diff = intrinsic - continuation
            
            # Find the boundary: the largest S where diff >= 0
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
        
        # Time check every 200 steps
        if step % 200 == 0:
            if time.time() > deadline - 10:
                # If running out of time, break and fill remaining boundaries
                # with conservative estimates
                for remaining_step in range(step - 1, 0, -1):
                    if remaining_step in exercise_steps:
                        idx = exercise_steps[remaining_step]
                        boundaries[idx] = K  # Conservative: exercise at K
                break

    # Boundary at maturity
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
