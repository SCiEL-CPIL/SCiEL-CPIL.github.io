"""Bermudan exercise policy using binomial tree backward induction."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Use a binomial tree with steps aligned to exercise dates
    # 12 exercise dates, monthly, so N should be a multiple of 12
    N = 1200  # 100 steps per month
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Exercise dates in step indices (1-indexed, no time-0 exercise)
    steps_per_month = N // n_exercise
    exercise_steps = [steps_per_month * (i + 1) for i in range(n_exercise)]
    
    # Build the binomial tree for stock prices
    # At step n, stock price is S0 * u^j * d^(n-j) for j = 0..n
    # We only need to store values at exercise dates
    
    # For a put option, continuation value = discount * (p * V_up + (1-p) * V_down)
    # Exercise value = max(K - S, 0)
    # We exercise if exercise_value >= continuation_value
    
    # Work backward from maturity
    # At maturity (step N), the option value is the payoff
    # At each exercise date, we take max(exercise_value, continuation_value)
    
    # To save memory, we only track values at exercise dates
    # At each non-exercise step, we just compute continuation
    
    # Let's work backward from the last exercise date
    # At the last exercise date (step N), value = max(K-S, 0)
    
    # For efficiency, we'll compute the critical stock price at each exercise date
    # where exercise becomes optimal
    
    # Initialize: at maturity, critical price is K (exercise if S <= K)
    # Work backward through each exercise date
    
    # We need to find the optimal exercise boundary
    # For a Bermudan put, the boundary is monotonically decreasing over time
    # (you're more willing to exercise early when S is lower)
    
    # Let's compute the value function at each exercise date using backward induction
    # with interpolation between tree nodes
    
    # Simplified approach: use the tree to compute the continuation value
    # at each exercise date for a range of stock prices
    
    # Number of nodes at each exercise date
    # At step n, there are n+1 nodes with stock prices S0 * u^j * d^(n-j)
    
    # For the last exercise date (step N), the value is simply max(K - S, 0)
    
    # Let's compute the critical boundary at each exercise date
    # by working backward
    
    # At step N (last exercise), boundary = K
    # At earlier steps, we need to compute continuation value
    
    # For a put, we exercise when S is below a critical threshold
    # Let's find these thresholds
    
    # Work backward from the last exercise date
    # At each exercise date, for each possible stock price node,
    # compute max(exercise_value, continuation_value)
    
    # To be efficient, we'll only compute at exercise dates and interpolate
    
    # Let's use a coarser tree for speed: 12 steps (one per month)
    # This is a 12-step binomial tree
    
    N_coarse = n_exercise
    dt_coarse = T / N_coarse
    u_coarse = np.exp(sigma * np.sqrt(dt_coarse))
    d_coarse = 1.0 / u_coarse
    p_coarse = (np.exp(r * dt_coarse) - d_coarse) / (u_coarse - d_coarse)
    discount_coarse = np.exp(-r * dt_coarse)
    
    # Build the coarse tree
    # At step i, stock prices are S0 * u^j * d^(i-j) for j = 0..i
    # We need to find the exercise boundary at each step
    
    # Start from the last step (maturity)
    # Value at maturity: V = max(K - S, 0)
    
    # Work backward
    # At each step, for each node, compute:
    # continuation = discount * (p * V_up + (1-p) * V_down)
    # exercise = max(K - S, 0)
    # V = max(exercise, continuation)
    # Exercise if exercise >= continuation
    
    # Let's compute the critical stock price at each step
    # where exercise becomes optimal
    
    # For a put, the critical price decreases as we move backward in time
    # (less time value, so more likely to exercise)
    
    # Compute values at each node of the coarse tree
    # V[i][j] = value at step i, node j (stock price S0 * u^j * d^(i-j))
    
    # Initialize at maturity (step N_coarse)
    max_nodes = N_coarse + 1
    V = np.zeros((N_coarse + 1, N_coarse + 1))
    
    # At maturity
    for j in range(N_coarse + 1):
        S = 100.0 * (u_coarse ** j) * (d_coarse ** (N_coarse - j))
        V[N_coarse][j] = max(K - S, 0.0)
    
    # Work backward
    for i in range(N_coarse - 1, -1, -1):
        for j in range(i + 1):
            S = 100.0 * (u_coarse ** j) * (d_coarse ** (i - j))
            V_up = V[i + 1][j + 1]
            V_down = V[i + 1][j]
            continuation = discount_coarse * (p_coarse * V_up + (1 - p_coarse) * V_down)
            exercise_val = max(K - S, 0.0)
            V[i][j] = max(exercise_val, continuation)
    
    # Now find the critical stock price at each exercise date
    # At step i, exercise is optimal for S <= critical_price[i]
    # We can find this by looking at where V[i][j] == exercise_val[i][j]
    
    # For the exercise policy, we need to determine for a given current state
    # and time, whether to exercise
    
    # The time is given in terms of the 12 exercise dates
    # We need to map the current time to the appropriate step in the tree
    
    # For the exercise function, we'll use the critical prices from the tree
    # and interpolate for stock prices between nodes
    
    # Extract critical prices at each exercise date
    critical_prices = np.zeros(n_exercise)
    for i in range(n_exercise):
        # At step i, find the highest stock price where exercise is optimal
        # V[i][j] = max(exercise, continuation)
        # Exercise is optimal when exercise >= continuation
        # For a put, this happens when S is low enough
        
        # Find the largest j where exercise is optimal
        # Actually, for a put, lower j means lower stock price (more likely to exercise)
        # Let's check each node
        
        # The critical price is the stock price where exercise_value = continuation_value
        # We can find this by interpolation
        
        # For simplicity, let's use the stock price of the highest node where exercise is optimal
        # Or interpolate between the last non-exercise and first exercise nodes
        
        best_j = 0
        for j in range(i + 1):
            S = 100.0 * (u_coarse ** j) * (d_coarse ** (i - j))
            V_up = V[i + 1][j + 1] if j + 1 <= i else 0
            V_down = V[i + 1][j] if j <= i else 0
            if j + 1 > i:
                V_up = 0
            if j > i:
                V_down = 0
            continuation = discount_coarse * (p_coarse * V_up + (1 - p_coarse) * V_down)
            exercise_val = max(K - S, 0.0)
            if exercise_val >= continuation:
                best_j = j
        
        # The critical price is at node best_j
        # But we might want to interpolate
        S_best = 100.0 * (u_coarse ** best_j) * (d_coarse ** (i - best_j))
        
        # Check if the next node (higher stock price) is also exercise
        if best_j + 1 <= i:
            S_next = 100.0 * (u_coarse ** (best_j + 1)) * (d_coarse ** (i - best_j - 1))
            V_up = V[i + 1][best_j + 2] if best_j + 2 <= i else 0
            V_down = V[i + 1][best_j + 1] if best_j + 1 <= i else 0
            if best_j + 2 > i:
                V_up = 0
            if best_j + 1 > i:
                V_down = 0
            continuation = discount_coarse * (p_coarse * V_up + (1 - p_coarse) * V_down)
            exercise_val = max(K - S_next, 0.0)
            if exercise_val < continuation:
                # Interpolate between S_best and S_next
                # The critical price is somewhere between them
                # For simplicity, use the midpoint or just S_best
                critical_prices[i] = S_best
            else:
                critical_prices[i] = S_next
        else:
            critical_prices[i] = S_best
    
    # Now create the exercise policy
    class ExercisePolicy:
        def __init__(self, critical_prices, n_exercise):
            self.critical_prices = critical_prices
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            # current_state: array of stock prices
            # time: current time (0 to T)
            # Return: boolean vector, True if should exercise
            
            # Map time to exercise date index
            # 12 exercise dates at t = T/12, 2T/12, ..., T
            # time is in [0, T]
            
            # Determine which exercise date we're at or approaching
            # The exercise dates are at t_i = (i+1) * T/n_exercise for i = 0..n_exercise-1
            
            # If time is exactly at an exercise date, use that date's critical price
            # If time is between exercise dates, we can't exercise (Bermudan only allows exercise at specified dates)
            
            # For Bermudan options, you can only exercise at the specified dates
            # So we need to check if the current time is at an exercise date
            
            # The time parameter likely indicates which exercise date we're at
            # Let's assume time is the fraction of T, and exercise dates are at i/n_exercise
            
            # Actually, for Bermudan, exercise is only allowed at specific dates
            # The time parameter tells us which exercise opportunity we're at
            
            # Let's assume time is in [0, 1] and exercise dates are at i/n_exercise
            # We can only exercise at these dates
            
            # Find the exercise date index
            # If time is close to an exercise date, use that critical price
            
            # For simplicity, let's assume the time parameter directly maps to the exercise date
            # time = i/n_exercise for the i-th exercise date
            
            # Determine the exercise date index
            # If time is 0, no exercise (time-0 is not an exercise date)
            # Exercise dates are at t = 1/n_exercise, 2/n_exercise, ..., n_exercise/n_exercise = 1
            
            # Map time to exercise date
            # If time is at an exercise date, use the corresponding critical price
            # Otherwise, don't exercise
            
            # Let's check if time is at an exercise date
            # Exercise dates: t_i = (i+1) / n_exercise for i = 0..n_exercise-1
            
            # For robustness, let's find the nearest exercise date
            # But for Bermudan, you can ONLY exercise at the specified dates
            
            # Let's assume the evaluator calls exercise() only at exercise dates
            # In that case, time will be one of the exercise date times
            
            # Map time to exercise date index
            # t_i = (i+1) / n_exercise
            # So i = time * n_exercise - 1
            
            # Handle edge cases
            if time <= 0 or time >= 1.0 + 1e-10:
                # Not at an exercise date (or at maturity, which is an exercise date)
                # Actually, time = 1.0 is the last exercise date
                if time >= 1.0 - 1e-10:
                    idx = n_exercise - 1
                else:
                    return np.zeros(len(current_state), dtype=bool)
            else:
                # Find the exercise date
                # time should be approximately (i+1)/n_exercise
                idx_float = time * n_exercise - 1
                idx = int(round(idx_float))
                idx = max(0, min(idx, n_exercise - 1))
            
            critical_price = self.critical_prices[idx]
            
            # Exercise if stock price <= critical price
            return current_state <= critical_price
    
    return ExercisePolicy(critical_prices, n_exercise)
# EVOLVE-BLOCK-END
