import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_exercise_dates = 12

    # Determine shape of training_paths
    # Expected: (N, num_steps + 1) where columns are t=0..t=12
    # or (N, num_steps) where columns are t=1..t=12
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Identify which columns correspond to exercise dates t=1..t=12
    if total_cols == num_exercise_dates + 1:
        # Columns 0..12, exercise at indices 1..12
        S_exercise = training_paths[:, 1:]  # shape (N, 12)
    elif total_cols == num_exercise_dates:
        # Columns 0..11 correspond to t=1..t=12
        S_exercise = training_paths[:, :]  # shape (N, 12)
    else:
        # Fallback: assume last 12 columns are exercise dates
        S_exercise = training_paths[:, -num_exercise_dates:]
    
    N_paths = S_exercise.shape[0]
    
    # Discount factors: discount from date k to date k-1
    # dt = T / 12
    dt = T / num_exercise_dates
    discount = np.exp(-r * dt)
    
    # Payoff at each exercise date: max(K - S, 0)
    payoff = np.maximum(K - S_exercise, 0.0)  # shape (N, 12)
    
    # Continuation values: work backwards from the last exercise date
    # V_{12} = payoff_{12}
    # V_k = max(payoff_k, discount * E[V_{k+1} | S_k])
    
    # For the continuation value, we need E[V_{k+1} | S_k]
    # We'll use a regression-based approach: for each date k, regress V_{k+1} on S_k
    # Then continuation_value_k = discount * predicted_V_{k+1}
    
    # Use a piecewise linear regression / interpolation approach
    # For better accuracy, use a finer grid
    
    # Number of grid points for interpolation
    n_grid = 20000
    # Critical range for the put: [0, 1.2 * K]
    s_max = 1.2 * K
    grid = np.linspace(0.0, s_max, n_grid)
    
    # Store continuation values for each date
    # V[k] = continuation value at date k (before exercise decision)
    # V[11] (last date, index 11 for t=12) = payoff at last date
    V = np.zeros((N_paths, num_exercise_dates))
    V[:, -1] = payoff[:, -1]
    
    # Exercise decisions
    exercise = np.zeros((N_paths, num_exercise_dates), dtype=bool)
    exercise[:, -1] = True  # Always exercise at the last date if in the money
    
    # Work backwards
    for k in range(num_exercise_dates - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # Regress V[:, k+1] on S_exercise[:, k]
        # Use a simple approach: bin the S values and compute conditional means
        # For better accuracy, use linear interpolation on a grid
        
        # Create bins based on the grid
        # For each path, find the bin index
        S_k = S_exercise[:, k]
        
        # Clip S to grid range for interpolation
        S_clipped = np.clip(S_k, 0.0, s_max)
        
        # Find the two nearest grid points for each S value
        # Use searchsorted for efficiency
        indices = np.searchsorted(grid, S_clipped)
        indices = np.clip(indices, 1, n_grid - 1)
        
        # Linear interpolation weights
        x0 = grid[indices - 1]
        x1 = grid[indices]
        w = (S_clipped - x0) / (x1 - x0 + 1e-12)
        
        # We need to compute E[V_{k+1} | S_k] at each grid point
        # Assign each path to the nearest grid point and accumulate
        # For efficiency, use a histogram-based approach
        
        # Assign each path to the grid bin it falls into
        bin_idx = indices  # This is the right bin index
        
        # Accumulate sums and counts for each grid point
        # For linear interpolation, we need to distribute each path's contribution
        # to the two adjacent grid points
        
        sum_left = np.zeros(n_grid)
        sum_right = np.zeros(n_grid)
        count_left = np.zeros(n_grid)
        count_right = np.zeros(n_grid)
        
        # Each path contributes to grid[indices-1] and grid[indices]
        # Weight for left point: (1 - w), for right point: w
        # But we want E[V|S] at each grid point, so we should accumulate
        # V values weighted by how close each path's S is to the grid point
        
        # Simpler approach: for each grid point, find paths whose S is in the 
        # interval [grid[i-1], grid[i]] or [grid[i], grid[i+1]] and compute weighted average
        
        # Even simpler: use np.bincount with weights
        # For each path, add V_{k+1} * (1-w) to sum_left[indices-1] and V_{k+1} * w to sum_right[indices]
        
        np.add.at(sum_left, indices - 1, V[:, k+1] * (1.0 - w))
        np.add.at(sum_right, indices, V[:, k+1] * w)
        np.add.at(count_left, indices - 1, 1.0 - w)
        np.add.at(count_right, indices, w)
        
        # Compute conditional expectation at each grid point
        # At grid point i, the conditional expectation is approximately
        # (sum_left[i] + sum_right[i]) / (count_left[i] + count_right[i])
        total_sum = sum_left + sum_right
        total_count = count_left + count_right
        
        # Where count is zero, use the nearest non-zero value
        cond_mean = np.zeros(n_grid)
        mask = total_count > 0
        cond_mean[mask] = total_sum[mask] / total_count[mask]
        
        # Fill in gaps by forward/backward fill
        if not mask.all():
            # Forward fill
            last_valid = 0
            for i in range(n_grid):
                if mask[i]:
                    last_valid = i
                else:
                    cond_mean[i] = cond_mean[last_valid]
            # Backward fill for the beginning
            next_valid = n_grid - 1
            for i in range(n_grid - 1, -1, -1):
                if mask[i]:
                    next_valid = i
                else:
                    cond_mean[i] = cond_mean[next_valid]
        
        # Continuation value at date k for each path
        # Interpolate cond_mean at S_k
        cont_val = (1.0 - w) * cond_mean[indices - 1] + w * cond_mean[indices]
        cont_val = discount * cont_val
        
        # Exercise decision: exercise if payoff > continuation value
        exercise[:, k] = payoff[:, k] >= cont_val
        V[:, k] = np.where(exercise[:, k], payoff[:, k], cont_val)
    
    # Now we have the exercise decisions for the training paths
    # We need to create a policy that can be applied to new states
    # The policy should be: exercise if S <= threshold_k for each date k
    # Find the exercise boundary for each date
    
    thresholds = np.zeros(num_exercise_dates)
    for k in range(num_exercise_dates):
        if time.time() > deadline:
            break
        # Find the maximum S where exercise is optimal
        # Sort by S and find the boundary
        S_k = S_exercise[:, k]
        ex = exercise[:, k]
        
        # If all exercise, threshold is s_max
        # If none exercise, threshold is 0
        if ex.all():
            thresholds[k] = s_max
        elif not ex.any():
            thresholds[k] = 0.0
        else:
            # Find the boundary: max S where exercise=True and min S where exercise=False
            # Sort indices by S
            sorted_idx = np.argsort(S_k)
            S_sorted = S_k[sorted_idx]
            ex_sorted = ex[sorted_idx]
            
            # Find the last index where exercise is True
            # and the first index where exercise is False after that
            # The threshold is between these
            # For simplicity, use the max S where exercise=True
            ex_indices = np.where(ex_sorted)[0]
            if len(ex_indices) > 0:
                thresholds[k] = S_sorted[ex_indices[-1]]
            else:
                thresholds[k] = 0.0
    
    # Create a simple policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_step):
            # current_state: array of spot prices
            # time_step: integer, 0-indexed for the 12 exercise dates
            # Return boolean array: True if exercise
            if time_step < 0 or time_step >= 12:
                return np.zeros(len(current_state), dtype=bool)
            return current_state <= self.thresholds[time_step]
    
    policy = ExercisePolicy(thresholds, K)
    return policy
