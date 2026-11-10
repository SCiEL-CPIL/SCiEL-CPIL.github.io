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

    n = n_exercises
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)

    # Convert training_paths to array: shape (n_paths, n_time_steps)
    # n_time_steps should be n_exercises + 1 (including t=0) or n_exercises
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_paths, n_steps = paths.shape

    # Determine number of exercise dates from paths
    # Typically paths have n_exercises + 1 columns (t=0 to t=T)
    # Exercise dates are at indices 1..n_exercises
    if n_steps > n_exercises:
        # Use the last n_exercises + 1 columns if available
        paths = paths[:, -n_exercises - 1:] if n_steps >= n_exercises + 1 else paths
    # Ensure we have at least n_exercises exercise dates
    n_steps = paths.shape[1]
    n_ex = min(n_exercises, n_steps - 1) if n_steps > 1 else n_exercises
    # Adjust n to match actual data
    n = n_ex
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)

    # --- Step 1: Fast grid-based backward induction for initial boundaries ---
    S_min = 5.0
    S_max = 300.0
    n_grid = 3000
    S_grid = np.geomspace(S_min, S_max, n_grid)

    V_next = np.maximum(K - S_grid, 0.0)
    boundaries = [0.0] * (n + 1)
    boundaries[n] = K

    for j in range(n - 1, 0, -1):
        if time.time() > deadline - 30.0:
            for jj in range(j, 0, -1):
                if boundaries[jj] == 0.0:
                    boundaries[jj] = K * 0.95
            break
        intrinsic = np.maximum(K - S_grid, 0.0)
        Su = np.clip(S_grid * u, S_grid[0], S_grid[-1])
        Sd = np.clip(S_grid * d, S_grid[0], S_grid[-1])
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        continuation = df * (p * V_up + (1 - p) * V_down)
        V_current = np.maximum(intrinsic, continuation)
        diff = intrinsic - continuation
        exercise_mask = diff >= 0
        if np.any(exercise_mask):
            idx = np.where(exercise_mask)[0]
            last_idx = idx[-1]
            if last_idx < n_grid - 1:
                diff0 = diff[last_idx]
                diff1 = diff[last_idx + 1]
                if diff0 - diff1 != 0:
                    t = diff0 / (diff0 - diff1)
                    t = max(0.0, min(1.0, t))
                    boundaries[j] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[j] = S_grid[last_idx]
            else:
                boundaries[j] = S_grid[-1]
        else:
            boundaries[j] = 0.0
        V_next = V_current

    for j in range(1, n + 1):
        if boundaries[j] == 0.0:
            boundaries[j] = K * 0.95

    # --- Step 2: Refine using Longstaff-Schwartz regression on training paths ---
    # For each exercise date j (1..n-1), fit a regression of continuation value
    # on features of S_j, then find the boundary where intrinsic = regression
    
    # We'll use the training paths to estimate the continuation value via LS
    # Features: S, S^2, log(S), and maybe S*(K-S) for put-specific shape
    # Use only in-the-money or near-the-money paths for better regression
    
    for j in range(n - 1, 0, -1):
        if time.time() > deadline - 20.0:
            break
            
        S_j = paths[:, j]  # Stock price at exercise date j
        S_next = paths[:, j + 1] if j + 1 < n_steps else S_j  # Next step
        
        # Intrinsic value at j
        intrinsic_j = np.maximum(K - S_j, 0.0)
        
        # Estimate continuation value using the current boundaries for steps > j
        # and the grid-based V for the immediate next step
        # Actually, let's use the grid-based V_next (which was computed in step 1)
        # to get a better continuation estimate, then refine with regression
        
        # For LS: we need the true continuation value. We'll use the grid-based
        # value as a target and fit a regression to smooth it.
        # Continuation from grid: interpolate V_next at S_j
        S_j_clipped = np.clip(S_j, S_grid[0], S_grid[-1])
        V_cont_grid = np.interp(S_j_clipped, S_grid, V_next)
        
        # Fit polynomial regression: V_cont = f(S_j)
        # Use features: 1, S, S^2, log(S), (K-S)^+ for put shape
        logS = np.log(S_j + 1e-10)
        X = np.column_stack([
            np.ones_like(S_j),
            S_j,
            S_j ** 2,
            logS,
            np.maximum(K - S_j, 0.0),
            np.maximum(K - S_j, 0.0) * logS
        ])
        
        # Weighted regression: weight in-the-money paths more heavily
        weights = np.ones_like(S_j)
        itm_mask = S_j < K * 1.1  # Near or in the money
        weights[itm_mask] = 2.0
        
        try:
            Xw = X * np.sqrt(weights)[:, None]
            yw = V_cont_grid * np.sqrt(weights)
            coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            V_cont_reg = X @ coeffs
            # Ensure non-negative
            V_cont_reg = np.maximum(V_cont_reg, 0.0)
        except:
            V_cont_reg = V_cont_grid
        
        # Find boundary where intrinsic = V_cont_reg
        # For each path, compare intrinsic vs continuation
        diff = intrinsic_j - V_cont_reg
        # The boundary is where diff changes sign from positive to negative
        # Sort by S_j and find the largest S where diff >= 0
        sort_idx = np.argsort(S_j)
        S_sorted = S_j[sort_idx]
        diff_sorted = diff[sort_idx]
        
        # Find last index where diff >= 0
        positive_mask = diff_sorted >= 0
        if np.any(positive_mask):
            last_pos = np.where(positive_mask)[0][-1]
            if last_pos < len(S_sorted) - 1:
                # Interpolate
                S0 = S_sorted[last_pos]
                S1 = S_sorted[last_pos + 1]
                d0 = diff_sorted[last_pos]
                d1 = diff_sorted[last_pos + 1]
                if d0 - d1 != 0:
                    t = d0 / (d0 - d1)
                    t = max(0.0, min(1.0, t))
                    new_boundary = S0 + t * (S1 - S0)
                else:
                    new_boundary = S0
            else:
                new_boundary = S_sorted[-1]
        else:
            new_boundary = 0.0
        
        # Blend with grid-based boundary for stability
        if new_boundary > 0:
            boundaries[j] = 0.7 * boundaries[j] + 0.3 * new_boundary
        else:
            boundaries[j] = K * 0.95

    # --- Step 3: Local optimization of boundaries using training paths ---
    # Maximize expected discounted payoff on training paths
    def expected_payoff(bnds):
        total = 0.0
        for j in range(n, 0, -1):
            S_j = paths[:, j]
            intrinsic = np.maximum(K - S_j, 0.0)
            exercise_mask = S_j <= bnds[j]
            payoff = np.where(exercise_mask, intrinsic, 0.0)
            disc = math.exp(-r * j * dt)
            total += disc * np.mean(payoff)
        return total
    
    # Simple coordinate ascent optimization
    best_bnds = boundaries[:]
    best_payoff = expected_payoff(best_bnds)
    
    for iteration in range(5):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for j in range(1, n):  # Don't optimize maturity boundary
            # Try small perturbations
            current = best_bnds[j]
            step = max(0.5, current * 0.01)
            for delta in [-step, -step/2, step/2, step]:
                trial = best_bnds[:]
                trial[j] = max(1.0, current + delta)
                payoff = expected_payoff(trial)
                if payoff > best_payoff + 1e-8:
                    best_bnds[j] = trial[j]
                    best_payoff = payoff
                    improved = True
        if not improved:
            break
    
    boundaries = best_bnds

    # Final fallback
    for j in range(1, n + 1):
        if boundaries[j] <= 0:
            boundaries[j] = K * 0.95

    class ExercisePolicy:
        def __init__(self, bnds, K_val):
            self.boundaries = bnds
            self.K = K_val

        def exercise(self, current_state, time_param):
            S_arr = np.asarray(current_state, dtype=np.float64)

            if isinstance(time_param, (int, float, np.integer, np.floating)):
                t = int(time_param)
                if t < 1:
                    t = 1
                elif t > 12:
                    t = 12
                return S_arr <= self.boundaries[t]
            else:
                t_arr = np.asarray(time_param)
                t_clipped = np.clip(t_arr, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in t_clipped])
                if S_arr.ndim == 1 and b.ndim == 0:
                    return S_arr <= float(b)
                elif S_arr.ndim == 1 and b.ndim == 1:
                    return S_arr <= b
                else:
                    return S_arr <= b

    return ExercisePolicy(boundaries, K)
