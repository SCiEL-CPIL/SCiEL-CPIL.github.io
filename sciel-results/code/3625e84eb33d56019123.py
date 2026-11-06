"""Bermudan exercise policy using Longstaff-Schwarz Monte Carlo."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 130.0  # safety margin
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Number of training paths
    n_paths = training_paths.shape[0] if hasattr(training_paths, 'shape') else len(training_paths)
    
    # Exercise dates: 12 monthly, no time-0
    # Times: T/12, 2T/12, ..., 12T/12 = T
    exercise_times = np.linspace(T / n_exercise, T, n_exercise)
    
    # Discount factors between consecutive exercise dates
    # From time t_i to t_{i+1}: dt = T/12
    dt_ex = T / n_exercise
    discount_step = np.exp(-r * dt_ex)
    
    # We'll use polynomial regression for continuation values
    # Features: [1, S, S^2, S^3, S^4, S^5] - degree 5 polynomial
    max_degree = 5
    n_features = max_degree + 1
    
    # Precompute feature matrices for each exercise date
    # training_paths shape: (n_paths, n_exercise + 1) presumably, or (n_paths, n_times)
    # Assume training_paths[:, i] gives the state at exercise date i (0-indexed)
    # So training_paths[:, 0] is at time T/12, training_paths[:, 11] is at time T
    
    # We need to do backward induction from last exercise date to first
    # At the last exercise date (maturity for the Bermudan), value = max(K - S, 0)
    # At earlier dates, value = max(exercise_value, E[discount * future_value])
    
    # Store the fitted coefficients for each exercise date
    # coeffs[i] gives the coefficients for the continuation value at exercise date i
    # (i.e., the expected discounted value from exercising optimally after date i)
    
    coeffs = [None] * n_exercise
    exercise_decisions = [None] * n_exercise
    
    # Start from the last exercise date (index n_exercise - 1)
    # At maturity, the option value is just the intrinsic value
    # For the second-to-last date, we regress the intrinsic value at maturity
    
    # Let's work backwards
    # future_values[i] = the value at exercise date i+1 (already computed)
    # We need to compute the expected discounted value at date i given state at date i
    
    # For the last date (index 11), the value is just intrinsic
    # For date 10, we regress intrinsic value at date 11 on state at date 10
    
    # Let's define:
    # S_i = training_paths[:, i] for i = 0, ..., 11
    # At date 11 (last), value = max(K - S_11, 0)
    # At date 10, continuation = E[discount * value_11 | S_10]
    # We approximate this by regressing value_11 on features of S_10
    
    # Backward pass
    # future_value_at_next = value at date i+1
    # For i from 10 down to 0:
    #   Regress future_value_at_next on features(S_i)
    #   continuation_i = predicted value
    #   exercise_i = (intrinsic_i > continuation_i)
    #   value_i = max(intrinsic_i, continuation_i)
    #   future_value_at_next = value_i (for next iteration)
    
    # Initialize with the terminal value
    S_next = training_paths[:, n_exercise - 1]  # S at last exercise date
    future_value = np.maximum(K - S_next, 0.0)
    
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > DEADLINE:
            break
        
        S_i = training_paths[:, i]
        intrinsic_i = np.maximum(K - S_i, 0.0)
        
        # Build feature matrix for S_i
        # Features: [1, S_i, S_i^2, ..., S_i^max_degree]
        X = np.column_stack([S_i ** d for d in range(n_features)])
        
        # Regress future_value on X
        # We want: continuation_i ≈ E[discount * future_value | S_i]
        # So we regress (discount * future_value) on X
        target = discount_step * future_value
        
        # Use least squares
        try:
            coeffs_i, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            coeffs_i = np.zeros(n_features)
        
        # Predict continuation value
        continuation_i = X @ coeffs_i
        
        # Exercise decision
        exercise_i = intrinsic_i > continuation_i
        
        # Value at this date
        value_i = np.maximum(intrinsic_i, continuation_i)
        
        coeffs[i] = coeffs_i
        exercise_decisions[i] = exercise_i
        
        # Update for next iteration
        S_next = S_i
        future_value = value_i
    
    # Now we have coeffs[i] for i = 0, ..., n_exercise-2
    # For i = n_exercise-1, the value is just intrinsic (no continuation)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self):
            self.coeffs = coeffs
            self.n_features = n_features
            self.exercise_times = exercise_times
            self.n_exercise = n_exercise
            self.K = K
            self.T = T
            self.dt_ex = dt_ex
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (should be one of the exercise dates)
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date this is
            # time should be close to one of exercise_times
            # Find the closest exercise index
            idx = np.argmin(np.abs(self.exercise_times - time))
            
            # If this is the last exercise date, always exercise if in the money
            if idx == self.n_exercise - 1:
                return current_state < self.K
            
            # Otherwise, use the fitted coefficients
            if self.coeffs[idx] is None:
                # Fallback: exercise if in the money
                return current_state < self.K
            
            S = np.asarray(current_state, dtype=float)
            X = np.column_stack([S ** d for d in range(self.n_features)])
            continuation = X @ self.coeffs[idx]
            intrinsic = np.maximum(self.K - S, 0.0)
            
            return intrinsic > continuation
    
    policy = ExercisePolicy()
    return policy

# EVOLVE-BLOCK-END
