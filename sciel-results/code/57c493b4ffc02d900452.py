"""Bermudan put exercise policy via path-based AOMC with Longstaff-Schwartz regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # Get training paths
    # training_paths shape: (n_paths, n_time_steps) or similar
    # We need to understand the structure
    n_paths = training_paths.shape[0]
    
    # The training_paths likely contains spot prices at each exercise date
    # Let's assume shape is (n_paths, n_exercise) where each column is the spot at exercise date i
    # Or it could be (n_paths, n_time_steps) where n_time_steps > n_exercise
    
    # Let's check the shape
    if training_paths.ndim == 2:
        n_cols = training_paths.shape[1]
        # If n_cols == n_exercise, then each column is an exercise date
        # If n_cols > n_exercise, we need to subsample
        if n_cols >= n_exercise:
            # Subsample to get exercise dates
            indices = np.linspace(0, n_cols - 1, n_exercise).astype(int)
            exercise_spots = training_paths[:, indices]
        else:
            exercise_spots = training_paths
    else:
        exercise_spots = training_paths
    
    # Ensure we have the right number of columns
    if exercise_spots.shape[1] < n_exercise:
        # Pad or handle differently
        pass
    
    # Longstaff-Schwartz style regression for continuation values
    # Work backwards from the last exercise date
    
    # For a put option:
    # Payoff at exercise = max(K - S, 0)
    # Continuation value = expected discounted future value
    
    # We'll use polynomial regression for the continuation value
    # Features: S, S^2, S^3, 1/S, etc.
    
    def make_basis(S):
        """Create basis functions for regression."""
        S = np.asarray(S)
        return np.column_stack([
            np.ones_like(S),
            S,
            S**2,
            S**3,
            1.0 / (S + 1e-10),
            np.log(S + 1e-10),
        ])
    
    # Initialize continuation values at the last exercise date
    # At the last exercise date, continuation = 0 (no more exercise opportunities)
    # So the value is just the immediate payoff
    
    # V[i] will store the continuation value for each path at exercise date i
    # V has shape (n_paths,) and we update it backwards
    
    # Start from the last exercise date
    # At exercise date n_exercise-1 (0-indexed), the value is max(K - S, 0)
    S_last = exercise_spots[:, -1]
    V = np.maximum(K - S_last, 0.0)
    
    # Now work backwards from exercise date n_exercise-2 to 0
    # At each step, we estimate the continuation value using regression on paths that didn't exercise
    
    # Discount factor for one step
    disc = np.exp(-r * dt)
    
    # Store the exercise decisions for each path at each date
    # exercise_decision[i, p] = True if path p exercises at date i
    exercise_decision = np.zeros((n_exercise, n_paths), dtype=bool)
    
    # Process backwards
    for i in range(n_exercise - 2, -1, -1):
        S_i = exercise_spots[:, i]
        immediate_payoff = np.maximum(K - S_i, 0.0)
        
        # Continuation value is the discounted V from the next step
        # But V already includes the optimal exercise decision from next step onwards
        continuation = disc * V
        
        # Paths that would exercise at this date
        exercise_mask = immediate_payoff > continuation
        
        # For regression, we use paths that did NOT exercise (continuation paths)
        cont_mask = ~exercise_mask
        n_cont = np.sum(cont_mask)
        
        if n_cont > 10:
            # Fit regression on continuation paths
            X_cont = make_basis(S_i[cont_mask])
            y_cont = continuation[cont_mask]
            
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_cont, y_cont, rcond=None)
                # Predict continuation for all paths
                X_all = make_basis(S_i)
                cont_pred = X_all @ coeffs
            except:
                # Fallback: use simple average or direct value
                cont_pred = continuation.copy()
        else:
            # Not enough continuation paths, use direct value
            cont_pred = continuation.copy()
        
        # Update exercise decision
        exercise_decision[i, :] = immediate_payoff > cont_pred
        
        # Update V for the next iteration (going backwards)
        # V at this step is max(immediate, continuation)
        V = np.where(exercise_decision[i, :], immediate_payoff, cont_pred)
    
    # Now we have the exercise decisions for each path at each date
    # We need to create a policy function that takes current_state and time
    # and returns a boolean vector indicating whether to exercise
    
    # The policy should be based on the learned thresholds
    # For each exercise date i, we can determine the threshold S* such that
    # exercise if S < S* (for a put option)
    
    # Let's estimate the threshold for each exercise date
    thresholds = np.zeros(n_exercise)
    for i in range(n_exercise):
        S_i = exercise_spots[:, i]
        ex_mask = exercise_decision[i, :]
        if np.sum(ex_mask) > 0 and np.sum(~ex_mask) > 0:
            # Threshold is approximately the maximum S among exercised paths
            # or the minimum S among non-exercised paths
            # Use a smoothed estimate
            ex_S = S_i[ex_mask]
            cont_S = S_i[~ex_mask]
            # Threshold is between max(ex_S) and min(cont_S)
            if len(ex_S) > 0 and len(cont_S) > 0:
                thresholds[i] = (np.max(ex_S) + np.min(cont_S)) / 2.0
            elif len(ex_S) > 0:
                thresholds[i] = np.max(ex_S)
            else:
                thresholds[i] = np.min(cont_S)
        else:
            # All exercise or none exercise
            if np.all(ex_mask):
                thresholds[i] = np.inf  # Always exercise
            else:
                thresholds[i] = 0.0  # Never exercise
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_exercise):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            current_state: array of current spot prices
            time: current time (should correspond to one of the exercise dates)
            Returns: boolean vector indicating whether to exercise
            """
            # Determine which exercise date we're at
            # time is in [0, T], exercise dates are at 1*dt, 2*dt, ..., n_exercise*dt
            dt = self.T / self.n_exercise
            
            # Find the exercise index
            # time should be close to one of the exercise dates
            idx = int(round(time / dt)) - 1
            idx = np.clip(idx, 0, self.n_exercise - 1)
            
            # For a put, exercise if S < threshold
            return np.asarray(current_state) < self.thresholds[idx]
    
    policy = ExercisePolicy(thresholds, K, r, sigma, T, n_exercise)
    return policy

# EVOLVE-BLOCK-END
