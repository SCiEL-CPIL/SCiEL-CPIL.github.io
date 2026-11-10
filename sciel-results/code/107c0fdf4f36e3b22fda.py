import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo
    Backward Induction with polynomial regression for continuation values.
    """
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    n_exercise = 12
    T = contract.get('T', 1.0)
    dt = T / n_exercise
    
    # training_paths is expected to be shape (N_paths, n_exercise)
    # If it has more columns, we select the exercise dates.
    paths = np.asarray(training_paths, dtype=float)
    
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    N, M = paths.shape
    
    # Select exercise date columns
    if M == n_exercise:
        exercise_paths = paths
    else:
        # If M > n_exercise, assume paths are generated at finer steps or include t=0.
        # We need to pick indices corresponding to t = 1*dt, ..., n_exercise*dt.
        # If M includes t=0, indices are 1..n_exercise.
        # If M is just the exercise dates, M should be n_exercise.
        # Let's assume if M != n_exercise, we take the last n_exercise columns?
        # Or interpolate?
        # Given the ambiguity, and that "12 monthly exercise dates" is explicit,
        # it is highly likely M=12.
        # If M > 12, we'll take indices [M-12, ..., M-1] as a fallback,
        # but this is risky.
        # Better: assume the input is aligned.
        if M >= n_exercise:
            # Take the last n_exercise columns, assuming they are the exercise dates
            # or that t=0 is excluded.
            exercise_paths = paths[:, M-n_exercise:M]
        else:
            raise ValueError("Not enough steps in training_paths")

    # Backward Induction
    # We iterate from the last exercise date (index n_exercise-1) to the first (index 0)
    
    # Store regression coefficients for each time step
    # Key: time value, Value: coefficients
    coeffs = {}
    
    # Initialize value at the last step
    # At maturity (or last exercise date), exercise if in the money
    S_next = exercise_paths[:, n_exercise - 1]
    V_next = np.maximum(K - S_next, 0.0)
    
    # Iterate backwards
    for i in range(n_exercise - 2, -1, -1):
        S_curr = exercise_paths[:, i]
        
        # Discount the next value
        disc_factor = np.exp(-r * dt)
        C_disc = disc_factor * V_next
        
        # Regression: Estimate E[C_disc | S_curr]
        # Use polynomial features of degree 3
        # X = [1, S, S^2, S^3]
        X = np.column_stack((np.ones_like(S_curr), S_curr, S_curr**2, S_curr**3))
        
        # Solve least squares
        # np.linalg.lstsq is robust
        beta, residuals, rank, s = np.linalg.lstsq(X, C_disc, rcond=None)
        
        # Estimated continuation value
        C_est = X @ beta
        
        # Immediate exercise payoff
        U_curr = np.maximum(K - S_curr, 0.0)
        
        # Optimal value at this step
        V_curr = np.maximum(U_curr, C_est)
        
        # Store coefficients for this time step
        # Time is (i+1) * dt
        t_val = (i + 1) * dt
        coeffs[t_val] = beta
        
        # Update for next iteration
        V_next = V_curr
        S_next = S_curr

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, coeffs, K, r, dt, n_exercise):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.dt = dt
            self.n_exercise = n_exercise
            # Pre-sort times for fast lookup
            self.times = sorted(coeffs.keys())
            self.times_arr = np.array(self.times)
            
        def exercise(self, current_state, time):
            """
            Returns a boolean vector indicating whether to exercise.
            current_state: array-like of spot prices
            time: scalar or array-like of times
            """
            S = np.asarray(current_state, dtype=float)
            original_shape = S.shape
            S_flat = S.ravel()
            
            # Handle time
            if np.isscalar(time):
                t_arr = np.full_like(S_flat, float(time))
            else:
                t_arr = np.asarray(time, dtype=float).ravel()
                if len(t_arr) != len(S_flat):
                    # If lengths don't match, assume scalar time
                    t_arr = np.full_like(S_flat, float(time))
            
            # Find nearest exercise time for each state
            # Since n_exercise is small (12), we can find the index efficiently
            # Using searchsorted on sorted times
            indices = np.searchsorted(self.times_arr, t_arr)
            
            # searchsorted returns the index where the value would be inserted.
            # We need the nearest neighbor.
            # For each index, check left and right neighbor.
            # Handle boundaries
            indices = np.clip(indices, 1, len(self.times_arr) - 1)
            
            # Compare distance to left and right
            diff_left = np.abs(t_arr - self.times_arr[indices - 1])
            diff_right = np.abs(t_arr - self.times_arr[indices])
            
            # If right is closer, use indices, else indices-1
            # Note: searchsorted returns the first index where arr[i] >= x.
            # So arr[indices-1] < x <= arr[indices] (roughly).
            # We want the closer one.
            
            # Create array of chosen indices
            chosen_indices = np.where(diff_left < diff_right, indices - 1, indices)
            
            # Get the corresponding time keys
            chosen_times = self.times_arr[chosen_indices]
            
            # Now compute decisions vectorized
            # We need to get the correct beta for each state.
            # This is tricky because beta is a 4D vector for each time.
            # We can't easily vectorize the dot product with different betas per row in a simple matrix multiplication
            # unless we stack the betas and use advanced indexing.
            
            # Stack betas: shape (n_exercise, 4)
            betas_stack = np.array([self.coeffs[t] for t in self.times])
            
            # Get betas for each state: shape (N_states, 4)
            betas = betas_stack[chosen_indices]
            
            # Compute features for all states: shape (N_states, 4)
            S_col = S_flat.reshape(-1, 1)
            X_flat = np.column_stack((np.ones_like(S_flat), S_flat, S_flat**2, S_flat**3))
            
            # Compute C_est: shape (N_states,)
            C_est = np.sum(X_flat * betas, axis=1)
            
            # Compute U
            U = np.maximum(self.K - S_flat, 0.0)
            
            # Decision
            decisions = (U > C_est)
            
            return decisions.reshape(original_shape)

    return ExercisePolicy(coeffs, K, r, dt, n_exercise)
