import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0 # Safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Use a fine grid for accurate boundary calculation
    # 2400 steps ensures the 12 exercise dates align with grid points (2400/12 = 200)
    n_fine_steps = 2400
    dt = T / n_fine_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    
    # Grid for S
    # Cover a wide range. For a put, low S is critical.
    S_min = 0.01
    S_max = 200.0
    n_grid = 4000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity (step n_fine_steps)
    V = np.maximum(K - S_grid, 0.0)
    
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K # At maturity, exercise if S <= K
    
    # Backward induction
    # Exercise dates are at fine steps: 200, 400, ..., 2400
    # We iterate backwards from n_fine_steps - 1 down to 1
    for k in range(n_fine_steps - 1, 0, -1):
        if time.time() > deadline:
            break
            
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V
        V_up = np.interp(Su, S_grid, V)
        V_down = np.interp(Sd, S_grid, V)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V = np.maximum(intrinsic, continuation)
        
        # Check if this step corresponds to an exercise date
        # Exercise dates are at indices: 200, 400, ..., 2400 (in forward direction)
        # In backward induction, we are at step k.
        # The exercise date index j (1..12) corresponds to fine step k = j * 200.
        # So if k is a multiple of 200, it's an exercise date.
        if k % 200 == 0:
            j = k // 200 # j is 1..11 (since k goes down to 1, max k is 2200 for j=11, 2400 is maturity handled separately)
            if 1 <= j <= n_exercises - 1:
                diff = intrinsic - continuation
                # Find the last index where diff > 0
                # diff is positive for small S, negative for large S
                exercise_indices = np.where(diff > 0)[0]
                
                if len(exercise_indices) > 0:
                    last_idx = exercise_indices[-1]
                    if last_idx < n_grid - 1:
                        d0 = diff[last_idx]
                        d1 = diff[last_idx + 1]
                        if d1 != d0:
                            t_interp = d0 / (d0 - d1)
                            boundaries[j] = S_grid[last_idx] + t_interp * (S_grid[last_idx + 1] - S_grid[last_idx])
                        else:
                            boundaries[j] = S_grid[last_idx]
                    else:
                        boundaries[j] = S_grid[-1]
                else:
                    boundaries[j] = S_min

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
            if isinstance(time, (int, float)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                times = np.asarray(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                # Vectorized boundary lookup
                b = np.array([self.boundaries[t] for t in times_clipped])
                return current_state <= b

    return ExercisePolicy(boundaries, K)
