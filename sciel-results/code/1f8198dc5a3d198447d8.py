"""Bermudan exercise-policy research: Longstaff-Schwartz Monte Carlo with polynomial regression."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Exercise times: T/12, 2T/12, ..., T (no time-0 exercise)
    dt = T / n_dates
    exercise_times = np.linspace(dt, T, n_dates)
    
    # Training paths shape: (n_paths, n_dates+1) or (n_dates+1, n_paths)
    # Let's figure out the shape
    if training_paths.ndim == 2:
        if training_paths.shape[0] >= training_paths.shape[1]:
            # (n_paths, n_times)
            n_paths = training_paths.shape[0]
            n_times = training_paths.shape[1]
        else:
            # (n_times, n_paths)
            n_paths = training_paths.shape[1]
            n_times = training_paths.shape[0]
    else:
        raise ValueError("training_paths must be 2D")
    
    # Ensure we have the right number of time steps
    # Exercise dates are at indices 1, 2, ..., n_dates (0-indexed, with 0 being time 0)
    # So we need n_dates + 1 time steps
    if n_times != n_dates + 1:
        # Try to handle if the paths are already at exercise dates only
        # But the contract says 12 monthly exercise dates, so we expect 13 time points (0 to 12)
        pass
    
    # Extract spot prices at each exercise date
    # spots[i] is the spot price at exercise date i (0-indexed, i=0 is first exercise date at t=dt)
    if n_times == n_dates + 1:
        # paths[:, 1] is first exercise date, paths[:, 2] is second, etc.
        spots = [training_paths[:, i+1] for i in range(n_dates)]
    elif n_times == n_dates:
        # paths are only at exercise dates
        spots = [training_paths[:, i] for i in range(n_dates)]
    else:
        # Fallback: assume the last n_dates columns are exercise dates
        spots = [training_paths[:, i] for i in range(n_times - n_dates, n_times)]
    
    n_paths = len(spots[0])
    
    # Initialize continuation value at the last exercise date to 0
    # V[i] is the continuation value (value of holding) at exercise date i for each path
    V = np.zeros(n_paths)
    
    # Store the regression coefficients for each exercise date
    # We'll work backwards from the last exercise date to the first
    # For each date i (from n_dates-1 down to 0):
    #   Exercise payoff = max(K - S_i, 0)
    #   Continuation value = discounted expected V[i+1]
    #   We regress V[i+1] on S_i (and possibly other features) to estimate E[V[i+1] | S_i]
    #   Then continuation_value_i = exp(-r*dt) * regressed_value
    #   Exercise if payoff > continuation_value
    
    # For better results, use polynomial features and possibly add time as a feature
    # Since we're at a fixed time step, we can use polynomial regression on S
    
    # Let's use degree 3 polynomial regression for better approximation
    degree = 3
    
    # We'll store the decision at each date for the policy
    # But the policy needs to be state-based, so we need to store the regression model
    
    # Actually, for the exercise policy, we need a function that takes current_state and time
    # and returns a boolean. So we need to store the regression coefficients for each date.
    
    # Let's store the regression models
    regression_models = [None] * n_dates
    
    # Work backwards
    for i in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time left, break
            break
        
        S_i = spots[i]
        
        # Exercise payoff
        payoff = np.maximum(K - S_i, 0.0)
        
        # If this is the last exercise date, continuation value is 0
        if i == n_dates - 1:
            continuation = np.zeros(n_paths)
        else:
            # Continuation value is discounted expected future value
            # We need to estimate E[V[i+1] | S_i]
            # Use polynomial regression
            
            # Create polynomial features
            # Use S_i as the feature
            X = np.column_stack([
                np.ones(n_paths),
                S_i,
                S_i**2,
                S_i**3
            ])
            
            # We want to regress V[i+1] on X
            # Use least squares
            try:
                # Solve using lstsq for numerical stability
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, V, rcond=None)
                regressed_V = X @ coeffs
                # Clip to non-negative (value should be non-negative)
                regressed_V = np.maximum(regressed_V, 0.0)
                continuation = np.exp(-r * dt) * regressed_V
            except:
                # Fallback: use simple mean
                continuation = np.exp(-r * dt) * np.mean(V)
        
        # Exercise if payoff > continuation
        exercise_mask = payoff > continuation
        
        # Update V: if we exercise, we get the payoff; otherwise, we hold and get continuation
        V = np.where(exercise_mask, payoff, continuation)
        
        # Store the regression coefficients for this date
        if i < n_dates - 1:
            regression_models[i] = coeffs
        else:
            regression_models[i] = None  # Last date, no continuation
    
    # Now we need to create a policy object
    # The policy needs to exercise(current_state, time)
    # current_state is the spot price, time is the current time
    
    # We need to determine which exercise date we're at based on time
    # And then use the regression model for that date to decide whether to exercise
    
    class BermudanExercisePolicy:
        def __init__(self, regression_models, exercise_times, K, r, dt):
            self.regression_models = regression_models
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.dt = dt
            self.n_dates = len(exercise_times)
        
        def exercise(self, current_state, time):
            """
            current_state: spot price (scalar or array)
            time: current time (scalar or array)
            Returns: boolean vector indicating whether to exercise
            """
            # Determine which exercise date we're at
            # time should be one of the exercise_times
            
            # Handle scalar or array
            scalar_input = np.isscalar(current_state) or (isinstance(current_state, np.ndarray) and current_state.ndim == 0)
            
            if scalar_input:
                current_state = np.array([current_state])
                time = np.array([time])
            
            n = len(current_state)
            result = np.zeros(n, dtype=bool)
            
            for idx in range(n):
                t = time[idx]
                S = current_state[idx]
                
                # Find the exercise date index
                # time should match one of exercise_times
                # Use tolerance for floating point
                min_diff = np.min(np.abs(self.exercise_times - t))
                if min_diff > 1e-6:
                    # Not at an exercise date, don't exercise
                    result[idx] = False
                    continue
                
                date_idx = np.argmin(np.abs(self.exercise_times - t))
                
                # Exercise payoff
                payoff = max(self.K - S, 0.0)
                
                # Continuation value
                if date_idx == self.n_dates - 1:
                    continuation = 0.0
                else:
                    coeffs = self.regression_models[date_idx]
                    if coeffs is None:
                        continuation = 0.0
                    else:
                        # Compute regressed value
                        regressed_V = (coeffs[0] + coeffs[1]*S + coeffs[2]*S**2 + coeffs[3]*S**3)
                        regressed_V = max(regressed_V, 0.0)
                        continuation = np.exp(-self.r * self.dt) * regressed_V
                
                result[idx] = payoff > continuation
            
            if scalar_input:
                return result[0]
            return result
    
    return BermudanExercisePolicy(regression_models, exercise_times, K, r, dt)
# EVOLVE-BLOCK-END
