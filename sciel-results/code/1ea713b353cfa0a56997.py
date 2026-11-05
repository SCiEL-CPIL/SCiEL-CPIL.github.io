# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using high-resolution binomial tree backward induction.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_steps) or similar (not used directly for policy, 
                      but we use the contract parameters)
    - contract: dict with keys 'S0', 'K', 'r', 'sigma', 'T', 'n_exercise_dates', 'n_paths' etc.
    
    Returns:
    - An object with method exercise(current_state, time) returning a boolean vector.
    """
    t_start = _time.time()
    DEADLINE = 145.0  # safety margin under 150s wall-clock limit
    
    # Extract contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    # We use a fine binomial tree for backward induction
    # The reference uses 1200 steps; we'll use a similar or higher resolution
    # But we need to be careful about time. Let's use 1200 steps as in the reference.
    n_steps = 1200
    
    # Time step
    dt = T / n_steps
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_up = (np.exp(r * dt) - d) / (u - d)
    p_down = 1.0 - p_up
    discount = np.exp(-r * dt)
    
    # The exercise dates are at times T/12, 2*T/12, ..., 12*T/12 = T
    # So exercise occurs at steps: n_steps/12, 2*n_steps/12, ..., n_steps
    exercise_steps = [int((i + 1) * n_steps / n_exercise_dates) for i in range(n_exercise_dates)]
    # exercise_steps is a list of step indices where exercise can occur
    # The last one should be n_steps
    
    # We need to do backward induction from T to 0
    # At each node, the value is max(intrinsic, discounted expected future value)
    # But we only exercise at specific dates, so between exercise dates, it's just the expected value
    # (European-like)
    
    # We'll work on a grid of stock prices. Let's define the grid at each step.
    # At step j, the stock price is S0 * u^i * d^(j-i) for i = 0, 1, ..., j
    # But this creates a recombining tree. For simplicity and vectorization, we can use a 
    # uniform grid in log-space or linear space.
    
    # Actually, for a binomial tree, the standard approach is to use the tree structure.
    # But for vectorization, let's use a fixed grid.
    
    # Let's use a log-uniform grid from S_min to S_max
    # S_min: very small, S_max: large enough
    # The stock follows GBM, so over T=1 with sigma=0.2, the range is roughly S0 * exp(±3*sigma*sqrt(T))
    # Let's use a range that covers 99.9% of the distribution
    
    # Actually, for a put, we care about low stock prices. Let's set S_min very small and S_max large.
    # S_min = S0 * exp(-4 * sigma * sqrt(T)) 
    # S_max = S0 * exp(4 * sigma * sqrt(T))
    
    S_min = S0 * np.exp(-4 * sigma * np.sqrt(T))
    S_max = S0 * np.exp(4 * sigma * np.sqrt(T))
    
    # Number of grid points. Let's use 10000 for high resolution.
    n_grid = 10000
    
    # Create log-uniform grid
    log_S_min = np.log(S_min)
    log_S_max = np.log(S_max)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    grid = np.exp(log_grid)
    
    # Initialize value at maturity (step n_steps)
    # At maturity, the put payoff is max(K - S, 0)
    V_next = np.maximum(K - grid, 0.0)
    
    # We need to store the exercise boundaries at each exercise date
    # For each exercise date, we want the threshold: exercise if S <= threshold
    # Actually, for a put, we exercise when S is low enough. So the boundary is the highest S
    # at which it's optimal to exercise.
    
    # We'll do backward induction step by step
    # But we only need to track the value at exercise dates and compute boundaries there.
    
    # Let's do backward induction from step n_steps down to 0
    # At each step, if it's an exercise date, we take max(intrinsic, continuation)
    # Otherwise, we just take the continuation value
    
    # Map step to exercise date index
    # exercise_steps[k] is the step at which the k-th exercise date occurs (0-indexed)
    # Actually, exercise_steps[0] is the first exercise date (at T/12), etc.
    
    # Let's create an array that maps each step to whether it's an exercise date
    is_exercise_step = np.zeros(n_steps + 1, dtype=bool)
    for step in exercise_steps:
        if step <= n_steps:
            is_exercise_step[step] = True
    
    # Backward induction
    # We'll store the value function at each step? No, we only need the previous step's value.
    # But we need to compute the continuation value, which requires interpolation.
    
    # Actually, for a binomial tree on a fixed grid, we can use the fact that the tree recombines.
    # But our grid is fixed, so we need to interpolate the values from the next step's grid to the current step's grid.
    
    # Hmm, this is getting complicated. Let me think differently.
    
    # Alternative: use the binomial tree properly with the tree structure.
    # At step j, there are j+1 nodes. The stock price at node i is S0 * u^i * d^(j-i).
    # We can store the value at each node.
    
    # But j goes up to 1200, so we have 1201 nodes at the last step. That's manageable.
    
    # Let's do it the standard binomial tree way.
    
    # At step j, node i (i=0..j), stock price S_j_i = S0 * u^i * d^(j-i)
    # Value V_j_i
    
    # Start at step n_steps:
    # S_n_steps_i = S0 * u^i * d^(n_steps-i) for i=0..n_steps
    # V_n_steps_i = max(K - S_n_steps_i, 0)
    
    # Backward: for j from n_steps-1 down to 0:
    #   For each node i (0..j):
    #     S_j_i = S0 * u^i * d^(j-i)
    #     S_up = S_j_i * u = S0 * u^(i+1) * d^(j-i) = S_{j+1, i+1}
    #     S_down = S_j_i * d = S0 * u^i * d^(j-i+1) = S_{j+1, i}
    #     V_cont = discount * (p_up * V_{j+1, i+1} + p_down * V_{j+1, i})
    #     If is_exercise_step[j+1] (i.e., we're at an exercise date), then:
    #       V_j_i = max(K - S_j_i, V_cont)
    #     Else:
    #       V_j_i = V_cont
    
    # Wait, the exercise date is at step j+1, not j. Let me clarify.
    
    # Actually, the exercise dates are at times t_1, t_2, ..., t_12 where t_k = k*T/12.
    # In steps, these are at step k*n_steps/12.
    # So at step j, if j is one of the exercise steps, we can exercise.
    
    # Let me re-index: we go from step n_steps down to 0.
    # At step j, the time is j*dt.
    # If j is in exercise_steps, we can exercise.
    
    # Let's precompute the stock prices at each step.
    # But storing all stock prices for all steps would be 1200 * 1200 = 1.44M values, which is fine.
    
    # Actually, we can compute them on the fly.
    
    # Let's allocate memory for the value array.
    # V[j, i] for j=0..n_steps, i=0..j
    # We can use a 2D array with shape (n_steps+1, n_steps+1), but only the lower triangle is used.
    
    # For memory efficiency, we can just keep the next row and current row.
    
    V_next = np.zeros(n_steps + 1)
    
    # Initialize at step n_steps
    for i in range(n_steps + 1):
        S = S0 * (u ** i) * (d ** (n_steps - i))
        V_next[i] = max(K - S, 0.0)
    
    # We need to store the exercise boundary at each exercise date.
    # For each exercise date (step j), the boundary is the maximum S such that it's optimal to exercise.
    # Actually, for a put, we exercise when S <= B. So B is the threshold.
    # We can find B by finding the largest i such that V_j_i = K - S_j_i (i.e., exercise is optimal).
    
    boundaries = [None] * n_exercise_dates
    boundary_step_map = {step: idx for idx, step in enumerate(exercise_steps)}
    
    # Backward induction
    for j in range(n_steps - 1, -1, -1):
        V_curr = np.zeros(j + 1)
        for i in range(j + 1):
            S = S0 * (u ** i) * (d ** (j - i))
            # Get continuation value
            # V_{j+1, i} and V_{j+1, i+1}
            V_cont = discount * (p_up * V_next[i + 1] + p_down * V_next[i])
            
            if j in boundary_step_map:
                # Exercise date
                intrinsic = max(K - S, 0.0)
                V_curr[i] = max(intrinsic, V_cont)
            else:
                V_curr[i] = V_cont
        
        V_next = V_curr
        
        # Check if we've hit the deadline
        if _time.time() - t_start > DEADLINE - 5.0:
            break
    
    # Now we need to extract the boundaries. But we didn't store them during the loop.
    # Let's redo the backward induction but store the boundaries.
    
    # Actually, let's restructure: do the backward induction and store the value at each exercise step.
    
    # Let me restart the backward induction with boundary extraction.
    
    t_start = _time.time()
    
    V_next = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        S = S0 * (u ** i) * (d ** (n_steps - i))
        V_next[i] = max(K - S, 0.0)
    
    boundaries = [None] * n_exercise_dates
    boundary_step_map = {step: idx for idx, step in enumerate(exercise_steps)}
    
    for j in range(n_steps - 1, -1, -1):
        V_curr = np.zeros(j + 1)
        for i in range(j + 1):
            S = S0 * (u ** i) * (d ** (j - i))
            V_cont = discount * (p_up * V_next[i + 1] + p_down * V_next[i])
            
            if j in boundary_step_map:
                intrinsic = max(K - S, 0.0)
                V_curr[i] = max(intrinsic, V_cont)
            else:
                V_curr[i] = V_cont
        
        V_next = V_curr
        
        if j in boundary_step_map:
            idx = boundary_step_map[j]
            # Find the boundary: the largest S such that exercise is optimal
            # Exercise is optimal when intrinsic >= continuation
            # So find the largest i such that K - S_j_i >= V_cont
            # But we don't have V_cont stored. Let's recompute or store it.
            
            # Actually, we can find the boundary by checking where the value switches from exercise to continuation.
            # For a put, at low S, we exercise; at high S, we continue.
            # So the boundary is the highest S where we exercise.
            
            # Let's find the largest i such that V_curr[i] == K - S_j_i (approximately)
            # Or more precisely, where intrinsic >= continuation.
            
            # Recompute for this step
            S_vals = np.array([S0 * (u ** i) * (d ** (j - i)) for i in range(j + 1)])
            intrinsic_vals = np.maximum(K - S_vals, 0.0)
            
            # Continuation values
            V_cont_vals = np.zeros(j + 1)
            for i in range(j + 1):
                V_cont_vals[i] = discount * (p_up * V_next[i + 1] + p_down * V_next[i])
            
            # Exercise is optimal where intrinsic >= continuation
            exercise_mask = intrinsic_vals >= V_cont_vals
            if np.any(exercise_mask):
                # Find the largest i where exercise is optimal
                exercise_indices = np.where(exercise_mask)[0]
                max_i = exercise_indices[-1]
                boundary = S_vals[max_i]
                boundaries[idx] = boundary
            else:
                boundaries[idx] = 0.0  # Never exercise
        
        if _time.time() - t_start > DEADLINE - 5.0:
            break
    
    # If any boundaries are None (due to timeout), set them to a reasonable default
    # For a put, a reasonable default boundary is K (exercise if S <= K)
    # But actually, for a Bermudan put, the boundary is typically below K.
    # Let's set missing boundaries to K as a conservative default.
    for i in range(n_exercise_dates):
        if boundaries[i] is None:
            boundaries[i] = K
    
    # Now, create the exercise policy
    # The policy needs to map (current_state, time) to a boolean vector.
    # time is the current time (0 to T).
    # We need to determine which exercise date we're at (or the next one).
    
    # Actually, the exercise method is called at each time step, and we need to decide whether to exercise.
    # The time parameter tells us the current time.
    
    # For a given time t, we find the exercise date index: the largest k such that t_k <= t.
    # But actually, we can only exercise at the exercise dates. So if t is not exactly an exercise date,
    # we should not exercise (or the policy is only called at exercise dates).
    
    # Let's assume the exercise method is called at each of the 12 exercise dates.
    # So time will be t_1, t_2, ..., t_12.
    
    # For time t, find the exercise date index k such that t = k*T/12.
    # Then use boundaries[k-1] (0-indexed).
    
    class ExercisePolicy:
        def __init__(self, boundaries, T, n_exercise_dates):
            self.boundaries = boundaries
            self.T = T
            self.n_exercise_dates = n_exercise_dates
            self.dt = T / n_exercise_dates
        
        def exercise(self, current_state, time):
            # current_state: array of stock prices
            # time: current time (scalar or array)
            # Return: boolean vector, True if exercise
            
            # Determine which exercise date this is
            # time should be one of t_1, ..., t_12
            # k = round(time / self.dt)
            # But time might be an array? Let's assume scalar.
            
            if isinstance(time, np.ndarray):
                # If time is an array, handle element-wise
                k_vals = np.round(time / self.dt).astype(int)
                # Clip to valid range
                k_vals = np.clip(k_vals, 1, self.n_exercise_dates)
                # For each element, use the corresponding boundary
                result = np.zeros(len(current_state), dtype=bool)
                for idx in range(len(current_state)):
                    k = k_vals[idx]
                    boundary = self.boundaries[k - 1]
                    result[idx] = current_state[idx] <= boundary
                return result
            else:
                k = int(round(time / self.dt))
                k = max(1, min(k, self.n_exercise_dates))
                boundary = self.boundaries[k - 1]
                return current_state <= boundary
    
    return ExercisePolicy(boundaries, T, n_exercise_dates)

# EVOLVE-BLOCK-END
