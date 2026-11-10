import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

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

    # Use a finer grid, but with a transformation that concentrates points near the kink at S=K
    # For a put, the critical region is S < K. We'll use a log-spaced grid combined with linear near K.
    # Actually, let's use a hybrid: dense linear grid near [0, K] and sparser beyond.
    # But a simpler approach: use a grid that's denser near S=K.
    
    # Let's use a grid from S_min to S_max with non-uniform spacing.
    # We'll use a transformation: S = K * exp(x) where x is linearly spaced.
    # This gives log-uniform spacing, which is good for multiplicative processes.
    
    # But we need to be careful: the boundary is a function of S, and we want to resolve it well.
    # Let's try a combination: use a log grid but with enough points.
    
    # Focus grid on the region where the boundary is likely to be.
    # For a put, boundary is <= K. Likely in [0.5*K, K].
    # Use a denser grid in this region.
    S_min = 0.1
    S_max = 150.0
    n_grid = 50000  # Increase resolution significantly
    
    # Use log-spaced grid
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # V at maturity (step 12)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # boundaries[i] for i in 1..12
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K  # At maturity, exercise if S <= K
    
    # Backward induction from step 11 down to 1
    for j in range(n_exercises - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break: return what we have.
            # For missing boundaries, use a reasonable default
            for jj in range(j, 0, -1):
                boundaries[jj] = boundaries[n_exercises]  # Use maturity boundary as default
            break
            
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        # Find boundary
        diff = intrinsic - continuation
        
        # diff is positive for small S, negative for large S.
        # Find the index where sign changes.
        exercise_indices = np.where(diff > 0)[0]
        
        if len(exercise_indices) > 0:
            last_idx = exercise_indices[-1]
            if last_idx < n_grid - 1:
                # Linear interpolation for the boundary
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
            boundaries[j] = S_min  # Should not happen for a put with these params
        
        V_next = V_current
    
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
