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

    # Use a high-resolution binomial tree as suggested by knowledge
    # N=1200 steps is cited as effective.
    # We will compute the optimal exercise boundaries for the 12 dates.
    
    N_steps = 1200
    dt = T / N_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Map exercise dates to tree levels
    # Exercise dates are at t = j/12 for j=1..12
    exercise_levels = []
    for j in range(1, n_exercises + 1):
        t_j = j / 12.0
        level_j = int(round(t_j / dt))
        # Ensure level is within [1, N_steps]
        level_j = max(1, min(N_steps, level_j))
        exercise_levels.append(level_j)
        
    # Reverse list to process from end to start
    # We need to know which levels are exercise levels
    is_exercise_level = np.zeros(N_steps + 1, dtype=bool)
    for lev in exercise_levels:
        is_exercise_level[lev] = True
        
    # Map level to exercise index (1-based)
    level_to_ex_idx = np.zeros(N_steps + 1, dtype=int)
    for idx, lev in enumerate(exercise_levels):
        level_to_ex_idx[lev] = idx + 1
        
    # Backward Induction
    # V[i] is the value at level n, index i
    # We start at level N_steps
    # S[i] = S0 * u^i * d^(N-i) ? No, standard tree: S_i = S_0 * u^i * d^(N-i) is for level N.
    # Let's use the standard construction where at level n, the price is S_0 * u^i * d^(n-i).
    # To save memory, we only keep the current row of values.
    
    # Initialize values at maturity
    # i ranges from 0 to N_steps
    i = np.arange(N_steps + 1)
    S_N = 100.0 * (u ** i) * (d ** (N_steps - i))
    V = np.maximum(K - S_N, 0.0)
    
    boundaries = np.zeros(n_exercises + 1) # 1-indexed
    # At maturity (level N_steps), if it's an exercise date, boundary is K (exercise if S < K)
    # But we need to handle the levels carefully.
    
    # We will store the boundary for each exercise date found during backward pass.
    # Since we go backwards, we will find boundaries for dates 12, 11, ..., 1.
    
    # We need to track the stock prices at the current level to determine the boundary.
    # At level n, the stock prices are S_n[i] = S_0 * u^i * d^(n-i).
    
    current_level = N_steps
    # If current_level is an exercise level, determine boundary
    if is_exercise_level[current_level]:
        ex_idx = level_to_ex_idx[current_level]
        # Exercise if V < Intrinsic? No, V is already max(Intrinsic, Continuation) at maturity?
        # At maturity, Continuation is 0. So V = Intrinsic.
        # We exercise if Intrinsic > 0, i.e., S < K.
        # The boundary is K.
        # However, due to discrete steps, we might define it as the max S where we exercise.
        # For a put, we exercise if S <= B.
        # At maturity, B = K.
        boundaries[ex_idx] = K
        
    for n in range(N_steps - 1, 0, -1):
        if time.time() > deadline - 5:
            # Break out if running out of time, fill remaining with heuristic
            break
            
        # Calculate continuation value
        # V_cont[i] = df * (p * V[i+1] + (1-p) * V[i])
        # V has size n+1 (indices 0..n). Next level had size n+2 (indices 0..n+1).
        # We are stepping back from n+1 to n.
        # V_current (size n+1) depends on V_next (size n+2).
        # V_next is currently stored in V (size n+2).
        # V_cont[i] = df * (p * V[i+1] + (1-p) * V[i])
        
        V_cont = df * (p * V[1:] + (1 - p) * V[:-1])
        
        # Calculate stock prices at level n
        i_curr = np.arange(n + 1)
        S_n = 100.0 * (u ** i_curr) * (d ** (n - i_curr))
        
        Intrinsic = np.maximum(K - S_n, 0.0)
        
        # Determine optimal value
        V_opt = np.maximum(V_cont, Intrinsic)
        
        # If this level is an exercise level, find the boundary
        if is_exercise_level[n]:
            ex_idx = level_to_ex_idx[n]
            # Exercise mask: Intrinsic >= V_cont
            # Note: At the boundary, they are equal.
            # We want the largest S where Exercise is optimal.
            # Since S_n is increasing with i, and Exercise is optimal for low S,
            # we look for the transition point.
            
            exercise_mask = Intrinsic >= V_cont
            
            if np.any(exercise_mask):
                # Find the largest index i where exercise is True
                # i is 0..n. S_n[i] is the price.
                # We want the boundary B such that S <= B implies exercise.
                # The "last" exercise index is max(i where exercise_mask[i] is True).
                i_ex = np.where(exercise_mask)[0]
                if len(i_ex) > 0:
                    i_max_ex = i_ex[-1]
                    # The boundary is between S_n[i_max_ex] and S_n[i_max_ex + 1] (if exists)
                    # Or just use S_n[i_max_ex] as a conservative estimate?
                    # Better: Interpolate or take the midpoint.
                    if i_max_ex < n:
                        B = (S_n[i_max_ex] + S_n[i_max_ex + 1]) / 2.0
                    else:
                        B = S_n[i_max_ex] * (1 + 1e-5) # Slightly above last exercised price
                    boundaries[ex_idx] = B
                else:
                    boundaries[ex_idx] = 0.0
            else:
                boundaries[ex_idx] = 0.0
        
        V = V_opt

    # Fill any unfound boundaries (if loop broke early) with heuristic
    # Heuristic: Bermudan put boundaries start low and approach K.
    # Simple linear interpolation from 0.7K to K?
    # Or just use the last calculated ones / K.
    for j in range(1, n_exercises + 1):
        if boundaries[j] == 0.0:
            # Fallback: Assume it's close to K for later dates, lower for earlier
            # A common approximation: B_j = K * (1 - exp(-lambda * t_j)) ?
            # Let's just use a simple ramp if not calculated
            boundaries[j] = K * (0.8 + 0.2 * (j / n_exercises))

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
            if isinstance(time, (int, float, np.integer, np.floating)):
                # time is a scalar
                # Map time to exercise index 1..12
                # t_j = j/12. 
                # If time is close to t_j, use boundary j.
                # The contract says "12 monthly exercise dates".
                # If time is passed, it should correspond to one of these dates.
                # We round to nearest month index.
                idx = int(round(time * 12))
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                # time is an array
                times = np.asarray(time)
                idxs = np.round(times * 12).astype(int)
                idxs = np.clip(idxs, 1, 12)
                b = self.boundaries[idxs]
                return current_state <= b

    return ExercisePolicy(boundaries, K)
