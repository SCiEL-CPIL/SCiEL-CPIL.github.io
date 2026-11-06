import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin
    
    # Parse contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Exercise dates
    exercise_dates = np.linspace(T/n_exercise, T, n_exercise)
    
    # training_paths shape: (n_paths, n_time_steps)
    n_paths, n_steps = training_paths.shape
    
    # Find indices corresponding to exercise dates
    # Assuming training_paths are generated with uniform time steps
    dt_train = T / (n_steps - 1) if n_steps > 1 else T
    exercise_indices = []
    for t in exercise_dates:
        idx = int(round(t / dt_train))
        if idx >= n_steps:
            idx = n_steps - 1
        exercise_indices.append(idx)
    
    # Extract states at exercise dates
    # S_k[j] = state of path j at exercise date k
    S_at_exercise = np.array([training_paths[:, exercise_indices[k]] for k in range(n_exercise)])
    # S_at_exercise shape: (n_exercise, n_paths)
    
    # Work backward
    # V[k] = continuation value at exercise date k (before exercise decision)
    # At the last exercise date, continuation = 0, so exercise if in-the-money
    # V[n_exercise-1] = 0 (no future after last exercise)
    # Exercise value at last date: max(K - S, 0)
    
    # For LSMC:
    # At step k (from last to first):
    #   forward_value[j] = discounted optimal value from step k+1 onwards
    #   Regress forward_value on S_at_exercise[k]
    #   continuation_value = regressed value
    #   exercise if max(K - S, 0) > continuation_value
    
    # Initialize
    # After the last exercise, there's no value
    # The "forward value" at step k is the value of the option if we continue past step k
    # i.e., it's the discounted optimal value from steps k+1 to n_exercise-1
    
    # Let's define:
    # opt_value[k, j] = optimal value from exercise date k onwards (including exercise at k)
    # We compute backward.
    
    # At the last exercise date (k = n_exercise-1):
    # opt_value[n_exercise-1, j] = max(K - S[n_exercise-1, j], 0)
    # (we always exercise if in-the-money, since there's no future)
    
    # For k = n_exercise-2 down to 0:
    #   The forward value (value if we don't exercise at k) is:
    #   forward_value[j] = exp(-r * dt_exercise) * opt_value[k+1, j]
    #   where dt_exercise = T/n_exercise
    #   
    #   Regress forward_value on S_at_exercise[k, j]
    #   continuation_value[j] = regressed value
    #   
    #   exercise_value[j] = max(K - S_at_exercise[k, j], 0)
    #   opt_value[k, j] = max(exercise_value[j], continuation_value[j])
    #   exercise_decision[k, j] = (exercise_value[j] > continuation_value[j])
    
    dt_exercise = T / n_exercise
    discount_factor = np.exp(-r * dt_exercise)
    
    opt_value = np.zeros((n_exercise, n_paths))
    exercise_decision = np.zeros((n_exercise, n_paths), dtype=bool)
    
    # Last exercise date
    S_last = S_at_exercise[n_exercise-1]
    opt_value[n_exercise-1] = np.maximum(K - S_last, 0)
    exercise_decision[n_exercise-1] = opt_value[n_exercise-1] > 0
    
    # Work backward
    for k in range(n_exercise-2, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        S_k = S_at_exercise[k]
        # Forward value: discounted optimal value from next exercise date
        forward_value = discount_factor * opt_value[k+1]
        
        # Regress forward_value on S_k
        # Use polynomial regression of degree 2 or 3
        # Add a constant term
        X = np.column_stack([np.ones(n_paths), S_k, S_k**2, S_k**3])
        
        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, forward_value, rcond=None)
            continuation_value = X @ coeffs
        except:
            continuation_value = np.mean(forward_value) * np.ones(n_paths)
        
        # Ensure continuation value is non-negative
        continuation_value = np.maximum(continuation_value, 0)
        
        exercise_value = np.maximum(K - S_k, 0)
        
        # Exercise if exercise value > continuation value
        exercise_decision[k] = exercise_value > continuation_value
        opt_value[k] = np.where(exercise_decision[k], exercise_value, continuation_value)
    
    # Now we have exercise_decision[k, j] for each exercise date and path
    # We need to create a policy that, given current_state and time, returns whether to exercise
    
    # The policy is: exercise at time t if S < boundary(t)
    # We can estimate the boundary from the training data
    
    # For each exercise date k, find the threshold:
    # boundary[k] = max S such that exercise_decision[k] is True
    # More precisely, find the S value where exercise_value = continuation_value
    
    # Let's estimate the boundary by finding the maximum S where we exercise
    boundaries = np.zeros(n_exercise)
    for k in range(n_exercise):
        if np.any(exercise_decision[k]):
            exercising_S = S_at_exercise[k][exercise_decision[k]]
            boundaries[k] = np.max(exercising_S)
        else:
            boundaries[k] = 0.0  # Never exercise
    
    # Interpolate boundaries to create a continuous function
    # For the exercise() method, we need to handle arbitrary time values
    
    # Create a function that, given time t, returns the boundary
    # We can use linear interpolation between exercise dates
    
    class ExercisePolicy:
        def __init__(self, exercise_dates, boundaries, K, r):
            self.exercise_dates = exercise_dates
            self.boundaries = boundaries
            self.K = K
            self.r = r
        
        def exercise(self, current_state, time):
            # current_state: array of states
            # time: scalar or array of times
            # Return: boolean array, True if exercise
            
            # Determine which exercise date we're at
            # The time should be one of the exercise dates
            # Find the closest exercise date
            if np.isscalar(time):
                times = np.array([time])
            else:
                times = np.array(time)
            
            results = np.zeros(len(current_state), dtype=bool)
            
            for i, t in enumerate(times):
                # Find the exercise date index
                # The time should match one of the exercise dates
                idx = np.argmin(np.abs(self.exercise_dates - t))
                boundary = self.boundaries[idx]
                results[i::len(times)] = current_state[i::len(times)] < boundary
            
            return results
    
    return ExercisePolicy(exercise_dates, boundaries, K, r)
