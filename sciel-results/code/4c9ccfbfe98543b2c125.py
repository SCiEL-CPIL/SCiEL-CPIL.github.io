import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
        t_start = time.time()
        deadline = t_start + 110.0 # Safety margin

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
        
        # Grid for S
        # We need to cover the range where the option might be in-the-money or near it.
        # For a put, S < K is relevant.
        # S can go up, but the boundary is on the low side.
        # Let's use a grid that is fine enough to capture the kink at S=K and the behavior near 0.
        S_min = 0.01
        S_max = 200.0
        n_grid = 50000 # Higher resolution for better boundary accuracy
        S_grid = np.linspace(S_min, S_max, n_grid)
        
        # V at maturity (step 12)
        # Note: The problem says "12 monthly exercise dates (no time-0 exercise)".
        # So steps are 1..12.
        # At step 12 (maturity), V = max(K - S, 0)
        V_next = np.maximum(K - S_grid, 0.0)
        
        # boundaries[i] for i in 1..12
        # At i=12, the policy is "exercise if S < K" (which is always true if ITM).
        # Actually, at maturity, you exercise if intrinsic > 0.
        # The "boundary" at maturity is K.
        boundaries = [0.0] * (n_exercises + 1)
        boundaries[n_exercises] = K # At maturity, exercise if S <= K
        
        # Backward induction from step 11 down to 1
        for j in range(n_exercises - 1, 0, -1):
            if time.time() > deadline:
                # Emergency break: return what we have. 
                # For missing boundaries, assume no exercise (boundary 0) 
                # or exercise (boundary inf)? 
                # Usually, not exercising is safer if we don't know? 
                # But for a put, not exercising when you should is bad. 
                # However, 10k points * 12 steps is very fast. This is unlikely to trigger.
                pass
            
            intrinsic = np.maximum(K - S_grid, 0.0)
            
            # Continuation value
            Su = S_grid * u
            Sd = S_grid * d
            
            # Interpolate V_next
            # np.interp requires xp to be increasing, which it is.
            # If Su or Sd is out of bounds, np.interp uses the endpoint values.
            # This is acceptable for our range.
            V_up = np.interp(Su, S_grid, V_next)
            V_down = np.interp(Sd, S_grid, V_next)
            
            continuation = df * (p * V_up + (1 - p) * V_down)
            
            V_current = np.maximum(intrinsic, continuation)
            
            # Find boundary
            # Exercise if intrinsic > continuation
            # diff = intrinsic - continuation
            # We want the largest S where diff > 0.
            diff = intrinsic - continuation
            
            # diff is positive for small S, negative for large S.
            # Find the index where sign changes.
            # Since diff is monotonic decreasing (usually), we can use np.where.
            # But to be safe, find the last index where diff > 0.
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
                boundaries[j] = S_min # Should not happen for a put with these params, but safe default
            
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
                    # Handle broadcasting if current_state is 2D? 
                    # Task says "returning a boolean vector", implying 1D input/output or batch processing.
                    # If current_state is (N,) and time is (N,), b is (N,).
                    return current_state <= b
        
        return ExercisePolicy(boundaries, K)
