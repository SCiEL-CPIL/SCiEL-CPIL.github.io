import numpy as np
import math
import time
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)

    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}

    S_min = 0.1
    S_max = 300.0
    n_grid = 5000
    S_grid = np.linspace(S_min, S_max, n_grid)

    V_next = np.maximum(K - S_grid, 0.0)

    boundaries = np.zeros(n_exercises + 1)

    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        Su = S_grid * u
        Sd = S_grid * d
        
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        V_current = np.maximum(intrinsic, continuation)
        
        if step in exercise_steps:
            idx = exercise_steps[step]
            diff = intrinsic - continuation
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
        
        if time.time() > deadline - 5:
            break

    boundaries[12] = K

    # Optimization using training paths
    # Extract boundaries for indices 1..12
    x0 = boundaries[1:13].copy()
    
    # Prepare training paths for MC
    # Assume training_paths is (n_paths, 12)
    if training_paths is not None and training_paths.ndim == 2:
        n_paths, n_cols = training_paths.shape
        if n_cols >= 12:
            # Use first 12 columns as exercise dates
            paths = training_paths[:, :12]
            
            # Precompute discount factors
            t_j = np.arange(1, 13) / 12.0
            disc = np.exp(-r * t_j)
            
            def mc_value(boundaries):
                # boundaries: array of 12
                total_payoff = 0.0
                active = np.ones(n_paths, dtype=bool)
                
                for j in range(12):
                    if not np.any(active):
                        break
                    S = paths[:, j]
                    B = boundaries[j]
                    exercise_mask = (S <= B) & active
                    if np.any(exercise_mask):
                        payoff = (K - S[exercise_mask])
                        total_payoff += disc[j] * np.sum(payoff)
                        active[exercise_mask] = False
                        
                return total_payoff / n_paths

            def negative_mc_value(boundaries):
                return -mc_value(boundaries)

            # Optimize
            try:
                # Use Nelder-Mead
                res = minimize(negative_mc_value, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                
                # Check if we improved
                val_opt = -res.fun
                val_init = mc_value(x0)
                
                if val_opt > val_init:
                    x_opt = res.x
                    boundaries[1:13] = x_opt
                else:
                    # Keep tree boundaries
                    pass
                    
            except Exception:
                pass

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(round(time * 12))
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                times = np.asarray(time)
                idxs = np.round(times * 12).astype(int)
                idxs = np.clip(idxs, 1, 12)
                b = self.boundaries[idxs]
                return current_state <= b

    return ExercisePolicy(boundaries, K)
