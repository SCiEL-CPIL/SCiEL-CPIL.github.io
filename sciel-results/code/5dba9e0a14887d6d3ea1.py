"""Rewritten program using direct Monte Carlo optimization of exercise thresholds."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using direct Monte Carlo optimization
    of 12 threshold parameters (one per exercise date).
    
    The policy at each exercise date t_i is: exercise if S_t < threshold_i.
    We optimize thresholds to maximize expected discounted payoff.
    """
    # Parse contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly dates, no time-0
    # T=1 year, 12 monthly dates => at t = 1/12, 2/12, ..., 12/12
    exercise_times = np.linspace(1.0/12.0, 1.0, n_exercise)
    
    # training_paths shape: (n_paths, n_time_steps)
    n_paths = training_paths.shape[0]
    n_time_steps = training_paths.shape[1]
    
    # Map exercise dates to column indices
    if n_time_steps > 1:
        dt = T / (n_time_steps - 1)
    else:
        dt = T
    
    # For each exercise time, find the nearest column index
    exercise_cols = []
    for t_ex in exercise_times:
        col = int(round(t_ex / dt))
        col = min(col, n_time_steps - 1)
        exercise_cols.append(col)
    
    # Precompute spot prices and payoffs at each exercise date
    S_ex = np.array([training_paths[:, col] for col in exercise_cols])  # (n_exercise, n_paths)
    payoffs = np.maximum(K - S_ex, 0.0)  # (n_exercise, n_paths)
    
    # Discount factors from each exercise date to the next
    # disc[i] = exp(-r * (t_{i+1} - t_i)) for i < n-1
    disc = np.zeros(n_exercise)
    for i in range(n_exercise - 1):
        dt_next = exercise_times[i+1] - exercise_times[i]
        disc[i] = np.exp(-r * dt_next)
    
    # Discount from time 0 to first exercise date
    disc_first = np.exp(-r * exercise_times[0])
    
    def compute_policy_value(thresholds):
        """
        Compute the expected discounted payoff for a given set of thresholds.
        Uses backward induction on all training paths.
        """
        n_p = n_paths
        
        # cont[i, j] = continuation value at exercise date i for path j
        cont = np.zeros((n_exercise, n_p))
        # cont[-1, :] = 0 (if we don't exercise at last date, option expires)
        
        # Backward induction
        for i in range(n_exercise - 2, -1, -1):
            S_next = S_ex[i+1]
            payoff_next = payoffs[i+1]
            
            # Decision at i+1: exercise if S < threshold
            exercise_next = S_next < thresholds[i+1]
            
            # Value at i+1 given the policy
            value_next = np.where(exercise_next, payoff_next, cont[i+1, :])
            
            # Discount back to i
            cont[i, :] = disc[i] * value_next
        
        # Value at first exercise date
        S_first = S_ex[0]
        payoff_first = payoffs[0]
        exercise_first = S_first < thresholds[0]
        value_first = np.where(exercise_first, payoff_first, cont[0, :])
        
        # Total value at time 0
        total_value = disc_first * np.mean(value_first)
        
        return total_value
    
    # Initialize thresholds - start with K as a reasonable initial guess
    thresholds = np.full(n_exercise, K)
    
    # Direct Monte Carlo optimization using coordinate descent with adaptive step size
    # Start with a larger step and decrease over iterations
    n_iter = 300
    step = 1.0  # initial step size
    step_decay = 0.95  # geometric decay factor
    min_step = 0.01
    
    for it in range(n_iter):
        improved = False
        
        # Try each threshold coordinate
        for i in range(n_exercise):
            val_curr = compute_policy_value(thresholds)
            
            # Try +step
            thresh_plus = thresholds.copy()
            thresh_plus[i] += step
            val_plus = compute_policy_value(thresh_plus)
            
            # Try -step
            thresh_minus = thresholds.copy()
            thresh_minus[i] -= step
            val_minus = compute_policy_value(thresh_minus)
            
            # Choose the best direction
            if val_plus > val_curr and val_plus >= val_minus:
                thresholds[i] += step
                improved = True
            elif val_minus > val_curr and val_minus >= val_plus:
                thresholds[i] -= step
                improved = True
        
        # Decay step size if no improvement
        if not improved:
            step *= step_decay
            if step < min_step:
                break
        else:
            # Slightly decay step size even when improving, to allow fine-tuning
            step = max(step * 0.98, min_step)
    
    # Final refinement with smaller steps
    fine_step = 0.05
    for it in range(100):
        improved = False
        for i in range(n_exercise):
            val_curr = compute_policy_value(thresholds)
            
            thresh_plus = thresholds.copy()
            thresh_plus[i] += fine_step
            val_plus = compute_policy_value(thresh_plus)
            
            thresh_minus = thresholds.copy()
            thresh_minus[i] -= fine_step
            val_minus = compute_policy_value(thresh_minus)
            
            if val_plus > val_curr and val_plus >= val_minus:
                thresholds[i] += fine_step
                improved = True
            elif val_minus > val_curr and val_minus >= val_plus:
                thresholds[i] -= fine_step
                improved = True
        
        if not improved:
            break
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.n_exercise = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of current spot prices (one per path)
            - time: current time (scalar)
            
            Returns:
            - boolean vector indicating whether to exercise for each path
            """
            t_val = float(time)
            
            # Find the exercise date index corresponding to the current time
            idx = np.argmin(np.abs(self.exercise_times - t_val))
            
            # Check if it's close enough to an exercise date
            if abs(self.exercise_times[idx] - t_val) > 0.01:
                # Not an exercise date, return all False
                return np.zeros(len(current_state), dtype=bool)
            
            # Exercise if S < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, exercise_times, K)
# EVOLVE-BLOCK-END
