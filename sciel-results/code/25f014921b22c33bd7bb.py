import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_ex = contract.get('n_exercise_dates', 12)
    
    # Exercise times: monthly, no time-0
    dt = T / n_ex
    exercise_times = np.arange(1, n_ex + 1) * dt
    
    # Grid parameters
    n_grid = 2000  # number of price grid points
    S_min = K * 0.3
    S_max = K * 1.8
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Risk-neutral parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Terminal payoff (European put at T)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Thresholds: exercise if S <= threshold
    thresholds = np.zeros(n_ex)
    
    # Backward induction
    for k in range(n_ex - 1, -1, -1):
        # Compute continuation value at S_grid
        # For each S_j, the next states are S_j * u and S_j * d
        S_up = S_grid * u
        S_down = S_grid * d
        
        # Interpolate V_next at S_up and S_down
        # Use linear interpolation
        V_up = np.interp(S_up, S_grid, V_next, left=0.0, right=0.0)
        V_down = np.interp(S_down, S_grid, V_next, left=0.0, right=0.0)
        
        cont_val = disc * (p_up * V_up + (1 - p_up) * V_down)
        
        # Exercise value
        ex_val = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_current = np.maximum(ex_val, cont_val)
        
        # Threshold: largest S where ex_val >= cont_val
        # Find where ex_val >= cont_val
        exercise_mask = ex_val >= cont_val
        if np.any(exercise_mask):
            # Find the largest index where exercise is optimal
            idx = np.max(np.where(exercise_mask)[0])
            thresholds[k] = S_grid[idx]
        else:
            thresholds[k] = 0.0
        
        V_next = V_current
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Create policy object
    class BermudanPolicy:
        def __init__(self, thresholds, exercise_times, K, r, sigma, T, n_ex):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_ex = n_ex
        
        def exercise(self, current_state, time_val):
            """
            current_state: array of stock prices
            time_val: current time (scalar)
            returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state)
            # Find which exercise period we're in
            # Exercise dates are at exercise_times[0], exercise_times[1], ...
            # If time_val is between exercise_times[k-1] and exercise_times[k], 
            # the next exercise opportunity is at exercise_times[k]
            # We should exercise if S < threshold for the next exercise date
            
            # Determine the next exercise index
            # If time_val < exercise_times[0], next ex is 0
            # If exercise_times[k-1] <= time_val < exercise_times[k], next ex is k
            # If time_val >= exercise_times[-1], we're at or after last exercise
            
            if time_val < self.exercise_times[0]:
                ex_idx = 0
            else:
                # Find the smallest k such that exercise_times[k] > time_val
                # If time_val is exactly at an exercise time, we can exercise now
                # The problem says "12 monthly exercise dates (no time-0 exercise)"
                # So at time t_k, we can exercise
                # If time_val >= exercise_times[k], the next available exercise is k+1 (if k < n_ex-1)
                # Actually, if time_val is exactly at exercise_times[k], we should use threshold[k]
                
                # Let's find the next exercise date >= time_val
                # If time_val is at or before an exercise date, use that date's threshold
                idx = np.searchsorted(self.exercise_times, time_val, side='right') - 1
                if idx < 0:
                    idx = 0
                elif idx >= self.n_ex:
                    idx = self.n_ex - 1
                
                # If time_val is very close to an exercise date, use that date
                # Otherwise, use the next upcoming exercise date
                # Actually, for a Bermudan option, you can only exercise AT the exercise dates.
                # The policy should decide whether to exercise at the NEXT exercise date.
                # So if current time is t, the next exercise date is the smallest exercise_times[k] >= t.
                # But if t is exactly at an exercise date, we exercise now.
                
                # Let me reconsider: the evaluator calls exercise(current_state, time) 
                # and expects a boolean. The time is the current time.
                # The policy should return True if it's optimal to exercise at the current time,
                # but you can only exercise at exercise dates.
                # So if time is not at an exercise date, return False.
                # If time is at an exercise date k, return S < threshold[k].
                
                # Check if time_val is close to an exercise date
                for k in range(self.n_ex):
                    if abs(time_val - self.exercise_times[k]) < 1e-6:
                        return current_state < self.thresholds[k]
                
                # Not at an exercise date, cannot exercise
                return np.zeros(len(current_state), dtype=bool)
        
        def __call__(self, current_state, time_val):
            return self.exercise(current_state, time_val)
    
    policy = BermudanPolicy(thresholds, exercise_times, K, r, sigma, T, n_ex)
    return policy
