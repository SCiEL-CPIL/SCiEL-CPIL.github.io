import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    dt = T / n_exercises
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    # Backward induction to find optimal exercise boundaries
    # At maturity (t_12 = T), the put value is max(K - S, 0)
    # At earlier dates, we compare intrinsic value to discounted expected continuation
    
    # We'll compute the critical boundary at each exercise date
    # For a put, we exercise when S <= B_i
    
    # Use a fine binomial tree for accuracy
    n_steps = 1200  # Fine tree for accurate boundaries
    dt_fine = T / n_steps
    u_fine = math.exp(sigma * math.sqrt(dt_fine))
    d_fine = 1.0 / u_fine
    p_fine = (math.exp(r * dt_fine) - d_fine) / (u_fine - d_fine)
    
    # But we only exercise at 12 specific dates
    # Map the 12 exercise dates to the fine tree
    exercise_steps = [int(i * n_steps / n_exercises) for i in range(1, n_exercises + 1)]
    
    # This is getting complex. Let me use a simpler approach.
    # Since we know the parameters, use a moderate binomial tree with exactly 12 steps
    # and compute the optimal policy via backward induction.
    
    # Actually, for the policy, we just need to know: at each exercise date i, what is the
    # critical stock price B_i such that we exercise if S <= B_i?
    
    # Let me use a higher-resolution tree for better accuracy
    n_fine = 1200
    dt_fine = T / n_fine
    u_f = math.exp(sigma * math.sqrt(dt_fine))
    d_f = 1.0 / u_f
    p_f = (math.exp(r * dt_fine) - d_f) / (u_f - d_f)
    
    # Exercise dates in terms of fine tree steps
    ex_steps = [round(i * n_fine / n_exercises) for i in range(1, n_exercises + 1)]
    # ex_steps = [100, 200, 300, ..., 1200]
    
    # Backward induction on the fine tree, but only allow exercise at ex_steps
    # V[j] = value at step j
    # Initialize at maturity
    # At step 1200 (maturity), V = max(K - S, 0) for all nodes
    
    # This is memory intensive. Let me think of a better way.
    
    # Alternative: Use the fact that for a Bermudan put, the optimal exercise boundary
    # can be approximated well with a moderate tree. Let me use 12 steps directly.
    
    # With 12 steps, the tree has 2^12 = 4096 leaves, which is manageable.
    # But the boundaries won't be as accurate as with a finer tree.
    
    # Let me use a hybrid: compute boundaries using a fine tree but only track
    # the critical boundary at each of the 12 exercise dates.
    
    # Actually, the simplest accurate approach: use backward induction with a fine tree
    # but only store the boundary at each exercise date.
    
    # For a put, at each node, the value is max(intrinsic, discounted_expected_continuation)
    # The boundary is where intrinsic = continuation
    
    # Let me just use a 12-step binomial tree. It should be sufficient for this task.
    
    n = 12
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Build the tree
    # S[j] at step j: S0 * u^(j-k) * d^k for k=0..j
    # We'll compute values backward
    
    # Value array: V[j][k] = value at step j, node k
    # At maturity (j=12): V[12][k] = max(K - S[12][k], 0)
    
    # S at step j, node k: S0 * u^(j-k) * d^k
    # But S0 varies (90, 100, 110). However, the exercise policy should be
    # independent of S0 in the sense that it's a function of current S and time.
    # The optimal policy is: exercise if S <= B_j at step j.
    # B_j is independent of S0 (it's a property of the option and market parameters).
    
    # So let me compute B_j for each exercise date j.
    
    # For a put, the continuation value at step j is:
    # C_j(S) = df * [p * E[V_{j+1} | up] + (1-p) * E[V_{j+1} | down]]
    # where V_{j+1} is the value at the next step.
    
    # The boundary B_j satisfies: K - B_j = C_j(B_j)
    
    # Let me compute this numerically.
    
    # Use a fine grid for S at each step
    # At step j, S ranges from S0 * d^j to S0 * u^j
    # But we need this to be independent of S0... 
    
    # Actually, the boundary B_j is in absolute terms, not relative to S0.
    # Let me just compute the optimal policy using backward induction on a
    # reasonable grid.
    
    # Simplified approach: use the 12-step tree and find the boundary at each step.
    
    # Let me compute the value function at each exercise date using backward induction.
    
    # At maturity (step 12), V(S) = max(K - S, 0)
    
    # At step 11, for a given S:
    # Intrinsic = max(K - S, 0)
    # Continuation = df * [p * V(S*u) + (1-p) * V(S*d)]
    # V(S) = max(Intrinsic, Continuation)
    
    # The boundary B_11 is where K - S = df * [p * V(S*u) + (1-p) * V(S*d)]
    
    # This is a fixed-point equation. Let me solve it numerically.
    
    # Actually, for a put, the value function is convex and the boundary can be
    # found by bisection.
    
    # Let me implement backward induction with a grid.
    
    # Grid for S
    S_min = 0.1
    S_max = 200.0
    n_grid = 1000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries at each step (from last to first)
    boundaries = [0.0] * (n + 1)  # boundaries[j] for j=0..n, but j=0 is time 0 (no exercise)
    
    # Backward induction from step 11 down to step 1
    for j in range(n - 1, 0, -1):
        # For each S in the grid, compute intrinsic and continuation
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation: need V_next at S*u and S*d
        # Interpolate V_next at S*u and S*d
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        # Find boundary: where intrinsic = continuation
        # intrinsic - continuation = 0
        diff = intrinsic - continuation
        # Find the largest S where diff >= 0 (we exercise when S is small)
        # diff is decreasing in S (intrinsic decreases, continuation may vary)
        
        # Find the crossing point
        # For a put, we exercise when S is small, so diff > 0 for small S
        # Find the last index where diff > 0
        exercise_mask = diff > 0
        if np.any(exercise_mask):
            # Find the boundary
            idx = np.where(exercise_mask)[0]
            last_idx = idx[-1]
            if last_idx < n_grid - 1:
                # Linear interpolation
                diff0 = diff[last_idx]
                diff1 = diff[last_idx + 1]
                if diff1 != diff0:
                    t = diff0 / (diff0 - diff1)
                    boundaries[j] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[j] = S_grid[last_idx]
            else:
                boundaries[j] = S_grid[-1]
        else:
            boundaries[j] = 0.0
        
        V_next = V_current
    
    # Now boundaries[1] through boundaries[12] are the critical prices
    # At exercise date i (i=1..12), exercise if S <= boundaries[i]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            # time: which exercise date (0-indexed or 1-indexed?)
            # Let's assume time is 0-indexed: 0 for first exercise date, 11 for last
            # Or time could be 1-12. Let me handle both.
            # The task says "12 monthly exercise dates (no time-0 exercise)"
            # So time likely ranges from 1 to 12 or 0 to 11
            
            # current_state: array of stock prices
            # Return: boolean vector, True if exercise
            
            # Determine the index
            # If time is a scalar, use it directly
            # If time is an array, use element-wise
            
            if isinstance(time, (int, float)):
                idx = int(time)
                # Map to 1-12
                if idx < 1:
                    idx = 1
                elif idx > 12:
                    idx = 12
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                # time is an array
                times = np.array(time)
                # Map to 1-12
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(boundaries, K)
