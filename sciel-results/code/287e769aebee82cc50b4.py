"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_dates)
    n_paths = training_paths.shape[0]
    
    # Monthly exercise dates: t_i = i/12 for i=1,...,12
    dt = T / n_dates
    discount_factors = np.exp(-r * np.arange(1, n_dates + 1) * dt)
    
    # We need to determine the optimal exercise policy using backward induction
    # with a quantile grid approach
    
    # Get all states at each exercise date
    # training_paths[i, j] is the state at date j for path i
    # Dates are 1-indexed in the array (index 0 to n_dates-1 correspond to dates 1 to n_dates)
    
    # Build a quantile grid for each date
    # Use the training data to define a grid of states
    
    n_grid = 200  # Number of grid points
    
    # For each date, create a quantile grid
    grids = []
    for j in range(n_dates):
        states = training_paths[:, j]
        q = np.linspace(0.005, 0.995, n_grid)
        grid = np.quantile(states, q)
        grids.append(grid)
    
    # Backward induction: start from the last date
    # At the last date, exercise if payoff > 0 (always exercise a put if in the money)
    # Value at date j: max(payoff, discount * E[value at j+1 | state at j])
    
    # We'll estimate the continuation value using the training paths
    # For each grid point at date j, find the average discounted value at date j+1
    
    # First, compute the terminal payoff at the last date
    # Payoff for a put: max(K - S, 0)
    
    # We'll work backwards from date n_dates-1 (last) to date 0 (first)
    
    # For the last date (index n_dates-1), the value is just the payoff
    # For earlier dates, we need to estimate E[payoff or continue | current state]
    
    # Approach: For each date j and each grid point s, 
    # find all training paths that are near s at date j,
    # then look at their states at date j+1,
    # and compute the average of the value function at date j+1
    
    # Value function at date j+1: for each state, max(payoff, continuation)
    # But we're doing backward induction, so we build up the value function
    
    # Let's define value[j][k] = optimal value at date j given state = grids[j][k]
    
    # Initialize: at the last date, value = payoff
    values = [None] * n_dates
    values[n_dates - 1] = np.maximum(K - grids[n_dates - 1], 0.0)
    
    # For earlier dates, we need to compute the continuation value
    # For each grid point at date j, find nearby training paths and average their values at date j+1
    
    # Precompute: for each date j, for each training path, find the nearest grid index at date j
    # This is expensive, so we'll use a simpler approach: for each grid point, find paths within a tolerance
    
    # Actually, let's use a different approach: for each training path, 
    # at each date j, find the nearest grid point and accumulate the value at j+1
    
    # This is like a regression: for each grid point at date j, average the values at date j+1 
    # for all paths that are close to that grid point at date j
    
    # Let's use binning: for each date j, bin the states into n_grid bins
    # Then for each bin, average the values at date j+1 for paths in that bin
    
    # Precompute bin indices for each date
    bin_indices = []
    for j in range(n_dates):
        states = training_paths[:, j]
        # Use searchsorted to find which bin each state falls into
        # The grid is the bin centers, so we need to find the nearest bin
        # Use np.digitize or searchsorted
        # Actually, let's use the quantile grid: each state maps to a bin based on its rank
        # Simpler: use np.searchsorted to find the interval, then assign to the nearest grid point
        
        # For efficiency, let's just use the grid as bin edges
        # states in [grid[k], grid[k+1]) go to bin k
        # But we need to handle the boundaries
        
        # Let's use a simple approach: for each state, find the nearest grid point
        # This is O(n_paths * n_grid) which might be slow
        # n_paths=4096, n_grid=200, n_dates=12: 4096*200*12 = ~10M operations, should be fine
        
        # Use broadcasting for speed
        # states[:, None] - grids[j][None, :] gives distances
        # But this is 4096 x 200 = 819K per date, 12 dates = ~10M, should be OK
        
        dists = np.abs(states[:, None] - grids[j][None, :])
        bins = np.argmin(dists, axis=1)
        bin_indices.append(bins)
    
    # Now do backward induction
    for j in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, use a simple heuristic
            break
        
        # For each grid point at date j, compute the average value at date j+1 for paths in that bin
        # First, we need the values at date j+1 for each training path
        # Find the nearest grid point at date j+1 for each path
        states_next = training_paths[:, j + 1]
        dists_next = np.abs(states_next[:, None] - grids[j + 1][None, :])
        bins_next = np.argmin(dists_next, axis=1)
        
        # Value at date j+1 for each path: max(payoff, continuation)
        # But we're doing backward induction, so at date j+1, the value is:
        # If j+1 is the last date: payoff
        # Otherwise: max(payoff, discount * continuation)
        # But we've already computed values[j+1] as the optimal value at date j+1 for each grid point
        # So for each path, its value at date j+1 is values[j+1][bins_next]
        
        path_values_next = values[j + 1][bins_next]
        
        # Now, for each grid point at date j, average the path_values_next for paths in that bin
        # Use np.bincount
        bin_idx_j = bin_indices[j]
        
        # For each bin, sum the values and count the paths
        bin_sums = np.bincount(bin_idx_j, weights=path_values_next, minlength=n_grid)
        bin_counts = np.bincount(bin_idx_j, minlength=n_grid)
        
        # Avoid division by zero
        bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
        avg_continuation = bin_sums / bin_counts_safe
        
        # For bins with no paths, use the average of neighboring bins or the overall average
        # For now, just use the overall average
        overall_avg = np.mean(path_values_next)
        empty_bins = bin_counts == 0
        avg_continuation[empty_bins] = overall_avg
        
        # Continuation value at date j: discount * avg_continuation
        discount = np.exp(-r * dt)
        continuation = discount * avg_continuation
        
        # Payoff at date j
        payoff = np.maximum(K - grids[j], 0.0)
        
        # Optimal value: max(payoff, continuation)
        values[j] = np.maximum(payoff, continuation)
    
    # Now we have the value function at each date
    # The exercise policy: exercise at date j if payoff >= continuation
    # We need to determine for each state whether to exercise
    
    # For the exercise function, given a current state and time, we need to return a boolean
    # The time corresponds to the exercise date index
    
    # Let's precompute the exercise decision for each grid point at each date
    exercise_decisions = [None] * n_dates
    
    for j in range(n_dates):
        if j == n_dates - 1:
            # Last date: exercise if in the money
            exercise_decisions[j] = grids[j] < K
        else:
            # Exercise if payoff >= continuation
            payoff = np.maximum(K - grids[j], 0.0)
            
            # Recompute continuation for this date
            states_next = training_paths[:, j + 1]
            dists_next = np.abs(states_next[:, None] - grids[j + 1][None, :])
            bins_next = np.argmin(dists_next, axis=1)
            path_values_next = values[j + 1][bins_next]
            
            bin_idx_j = bin_indices[j]
            bin_sums = np.bincount(bin_idx_j, weights=path_values_next, minlength=n_grid)
            bin_counts = np.bincount(bin_idx_j, minlength=n_grid)
            bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
            avg_continuation = bin_sums / bin_counts_safe
            overall_avg = np.mean(path_values_next)
            empty_bins = bin_counts == 0
            avg_continuation[empty_bins] = overall_avg
            
            discount = np.exp(-r * dt)
            continuation = discount * avg_continuation
            
            exercise_decisions[j] = payoff >= continuation
    
    # Now create the exercise policy object
    class ExercisePolicy:
        def __init__(self, grids, exercise_decisions, K, n_dates):
            self.grids = grids
            self.exercise_decisions = exercise_decisions
            self.K = K
            self.n_dates = n_dates
        
        def exercise(self, current_state, time):
            # current_state: array of states for multiple paths
            # time: the exercise date index (0-based, corresponding to date 1 to n_dates)
            # Return a boolean vector: True if exercise
            
            if isinstance(time, (int, float)):
                j = int(time)
            else:
                j = int(np.asarray(time).item())
            
            if j < 0 or j >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            grid = self.grids[j]
            decision = self.exercise_decisions[j]
            
            # For each state, find the nearest grid point and use its decision
            # current_state is an array
            dists = np.abs(current_state[:, None] - grid[None, :])
            nearest = np.argmin(dists, axis=1)
            
            return decision[nearest]
    
    return ExercisePolicy(grids, exercise_decisions, K, n_dates)
# EVOLVE-BLOCK-END
