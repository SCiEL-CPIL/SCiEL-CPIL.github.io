import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    dt = T / n_exercises
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # 1. Coarse Grid Initialization
    S_min = 0.0
    S_max = 150.0
    n_grid = 1000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    V_next = np.maximum(K - S_grid, 0.0)
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K
    
    for j in range(n_exercises - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        Su = S_grid * u
        Sd = S_grid * d
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        continuation = df * (p * V_up + (1 - p) * V_down)
        V_current = np.maximum(intrinsic, continuation)
        
        diff = intrinsic - continuation
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
        V_next = V_current

    # 2. Prepare Training Data for MC Optimization
    # Assume training_paths is (N, 13) with t=0 to t=12
    # If shape is (N, 12), it's t=1 to t=12
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    # Determine number of time steps in data
    # If 13 columns, indices 1..12 are exercise times.
    # If 12 columns, indices 0..11 correspond to exercise times 1..12.
    if paths.shape[1] == 13:
        S_ex = paths[:, 1:13] # Shape (N, 12)
    elif paths.shape[1] == 12:
        S_ex = paths # Shape (N, 12)
    else:
        # Fallback: assume last 12 columns if more, or just use what we have
        # But standard is 13 or 12.
        if paths.shape[1] > 12:
            S_ex = paths[:, -12:]
        else:
            S_ex = paths

    N_paths = S_ex.shape[0]
    # Discount factors for each exercise time
    t_ex = np.linspace(dt, T, n_exercises)
    disc_factors = np.exp(-r * t_ex)
    
    # Payoff at maturity if not exercised earlier is handled by the loop logic
    # We simulate forward.

    def evaluate_boundaries(bnd):
        # bnd: array of 12 boundaries
        # Returns mean discounted payoff
        payoffs = np.zeros(N_paths)
        exercised = np.zeros(N_paths, dtype=bool)
        
        for j in range(n_exercises):
            # If not yet exercised and S_j <= bnd[j], exercise
            mask = ~exercised & (S_ex[:, j] <= bnd[j])
            if np.any(mask):
                payoffs[mask] = (K - S_ex[mask, j]) * disc_factors[j]
                exercised[mask] = True
        
        # If not exercised, exercise at maturity (j=11 is the last step)
        # The loop above handles j=11. If not exercised by j=11, we must exercise at maturity.
        # Wait, the loop goes j=0 to 11 (indices for t_1 to t_12).
        # If not exercised at j=11 (t_12), it's maturity.
        # The condition `S_ex[:, 11] <= bnd[11]` might be false.
        # In Bermudan options, you MUST exercise at maturity.
        # So for any path not exercised by j=10, we check j=11.
        # If S_12 > bnd_11, we still exercise at maturity? 
        # Yes, at maturity, continuation value is 0, so intrinsic >= continuation.
        # So boundary at maturity is effectively infinity (or K).
        # My grid code set boundaries[12]=K, but for j=11 (index 10 in 0-based 1..12?), 
        # Let's check indices.
        # n_exercises = 12.
        # j in range(1, 13) in grid code.
        # boundaries[1] to boundaries[12].
        # In MC, bnd[0] corresponds to t_1, bnd[11] to t_12.
        # At t_12, we always exercise. So the boundary at t_12 should be treated as "always exercise".
        # So in the loop, for j=11 (last step), we should exercise regardless of boundary?
        # Or set bnd[11] to a very large number?
        # Actually, the optimal policy at maturity is always to exercise.
        # So we can just force exercise at j=11 for any path not exercised yet.
        
        # Let's refine the loop:
        payoffs = np.zeros(N_paths)
        exercised = np.zeros(N_paths, dtype=bool)
        
        for j in range(n_exercises - 1): # j=0 to 10 (t_1 to t_11)
            mask = ~exercised & (S_ex[:, j] <= bnd[j])
            if np.any(mask):
                payoffs[mask] = (K - S_ex[mask, j]) * disc_factors[j]
                exercised[mask] = True
        
        # Last step j=11 (t_12): Always exercise if not already
        mask_final = ~exercised
        if np.any(mask_final):
            payoffs[mask_final] = (K - S_ex[mask_final, 11]) * disc_factors[11]
            
        return np.mean(payoffs)

    # 3. Optimization Loop
    # Coordinate Ascent
    best_bnd = np.array(boundaries[1:13], dtype=np.float64)
    best_val = evaluate_boundaries(best_bnd)
    
    # Define search ranges for each boundary based on data or grid
    # Use a window around the current best
    step_sizes = np.ones(12) * 1.0
    
    max_iters = 10 # Number of full passes
    points_per_search = 11 # Points in 1D search
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        for j in range(12):
            if time.time() > deadline - 2.0:
                break
                
            # Define search range
            # Look at the distribution of S_ex[:, j]
            s_j = S_ex[:, j]
            s_min = np.percentile(s_j, 1)
            s_max = np.percentile(s_j, 99)
            
            # Center search around current best_bnd[j]
            center = best_bnd[j]
            # Range should cover likely optimal region
            # If center is far from data, clamp to data range
            lo = max(0.0, center - 5.0)
            hi = min(150.0, center + 5.0)
            
            # Ensure we search around the data points if possible
            # If the current boundary is outside the data range, the optimal might be at the edge of data
            # Let's just do a grid search in [lo, hi]
            candidates = np.linspace(lo, hi, points_per_search)
            
            best_local_val = best_val
            best_local_bnd = best_bnd[j]
            
            # We can vectorize the evaluation for these candidates?
            # evaluate_boundaries takes a full vector.
            # We can create a batch of candidate vectors.
            # But N_paths * 12 * 11 evaluations is fine.
            
            for c in candidates:
                trial_bnd = best_bnd.copy()
                trial_bnd[j] = c
                val = evaluate_boundaries(trial_bnd)
                if val > best_local_val:
                    best_local_val = val
                    best_local_bnd = c
            
            if best_local_bnd != best_bnd[j]:
                best_bnd[j] = best_local_bnd
                best_val = best_local_val
                improved = True
        
        if not improved:
            # Try smaller steps or break
            step_sizes *= 0.5
            if np.all(step_sizes < 0.1):
                break

    # Update boundaries list for the class
    boundaries[1:13] = best_bnd

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            if isinstance(time, (int, float)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                # At maturity (idx=12), always exercise?
                # The grid logic handled this.
                # If idx == 12, boundary is K (or higher).
                # If S <= K, exercise. If S > K, don't?
                # But at maturity, you MUST exercise.
                # So if idx == 12, return True?
                # The problem says "Bermudan put".
                # At maturity, it becomes European.
                # So if t == T, exercise if payoff > 0.
                # My boundary for j=12 is K.
                # So S <= K -> True. S > K -> False.
                # This is correct for a put.
                return current_state <= b
            else:
                times = np.asarray(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return current_state <= b

    return ExercisePolicy(boundaries, K)
