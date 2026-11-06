# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using backward induction on a quantile grid.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Extract parameters from contract
    # Based on task description: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # 12 monthly exercise dates
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_dates)
    # The task says "12 monthly exercise dates (no time-0 exercise)".
    # Usually training_paths includes time 0. Let's check dimensions.
    # If training_paths has shape (N, M), M is likely n_dates + 1 or n_dates.
    # The task says "12 monthly exercise dates".
    # Let's assume the provided paths align with the exercise dates.
    # If the contract specifies n_dates=12, and paths have 13 columns (t0..t12), 
    # we need to be careful.
    # However, the standard setup for such tasks usually provides paths for the exercise dates.
    # Let's inspect training_paths shape.
    
    n_paths, n_cols = training_paths.shape
    
    # Determine which columns correspond to exercise dates
    # If n_cols == n_dates, we assume it's just the exercise dates.
    # If n_cols == n_dates + 1, the first column is t=0 (no exercise) and rest are exercise dates.
    # The task says "no time-0 exercise".
    # Let's assume the standard case where training_paths contains prices at all time steps including t=0.
    # If n_cols > n_dates, we take the last n_dates columns? Or first?
    # Actually, for a Bermudan option, exercise is possible at discrete times.
    # Let's assume the columns in training_paths correspond to t_1, ..., t_n_dates if n_cols == n_dates.
    # If n_cols == n_dates + 1, t_0 is col 0, and t_1..t_n_dates are cols 1..n_dates.
    
    # Let's be robust.
    if n_cols >= n_dates:
        # Assume the last n_dates columns are the exercise dates?
        # Or if it's a GBM path, t=0 is the start.
        # Let's look at the "12 monthly exercise dates".
        # If T=1 and 12 monthly, dt=1/12.
        # If n_cols == 13, indices 1..12 are exercise dates.
        # If n_cols == 12, indices 0..11 are exercise dates?
        # Let's assume the provided paths are exactly the states at the exercise dates.
        # This is a common convention in these simplified contracts.
        exercise_data = training_paths[:, :n_dates] if n_cols == n_dates else training_paths[:, 1:n_dates+1]
    else:
        # Fallback: use all columns
        exercise_data = training_paths
    
    # Define the grid based on quantiles of the final time step or all steps?
    # Using all steps ensures coverage of the whole state space encountered.
    # However, backward induction requires a fixed grid.
    # A common approach is to use the quantiles of the final step (t_n) to define the grid,
    # as the distribution widens over time. But early steps are more concentrated.
    # A better grid is often constructed from the union of states or just the final step's quantiles
    # mapped back, but since we have the paths, we can use the quantiles of the *current* step
    # for each time step? No, the grid must be consistent for interpolation of continuation values.
    # Standard "Longstaff-Schwartz" or "Grid" methods use a fixed grid.
    # Let's use the quantiles of the final time step (t_n_dates) to define the grid points.
    # This grid will cover the range of prices at maturity.
    # Prices at earlier times are generally lower variance, so they will fall within this grid 
    # (mostly, unless there's a huge jump, but GBM is continuous).
    
    # Actually, a more robust grid for GBM is to use log-spaced points or quantiles of the *final* distribution.
    # Let's take 200 quantiles of the final column.
    n_grid_points = 200
    final_prices = exercise_data[:, -1]
    grid = np.quantile(final_prices, np.linspace(0.01, 0.99, n_grid_points))
    # Ensure grid is strictly increasing
    grid = np.unique(grid)
    
    # If grid is too small, expand
    if len(grid) < 10:
        grid = np.linspace(np.min(final_prices)*0.5, np.max(final_prices)*1.5, 50)
        
    # Backward Induction
    # V_t[S] is the value of the option at time t given state S.
    # Payoff_t[S] = max(K - S, 0)
    
    # Initialize value at the last exercise date (t_n_dates)
    # At the last date, it's an American put (exercise or hold to maturity? No, Bermudan ends there).
    # So V_{last} = Payoff_{last}
    
    # We need to map states to grid points.
    # For each time step t from last to first:
    # 1. Calculate intrinsic value at grid points.
    # 2. Calculate continuation value: discount expected value of V_{t+1} at next time step's states.
    #    Since we are on a grid, we need to estimate E[V_{t+1}(S_{t+1}) | S_t = grid_i].
    #    This is the hard part without a full PDE solver.
    #    Approximation: Use the training paths to estimate the conditional expectation.
    #    For each grid point g, find paths where S_t is close to g, and average V_{t+1}(S_{t+1}).
    
    # This "Monte Carlo regression" approach is computationally heavy if done naively for every grid point.
    # Alternative: Interpolate V_{t+1} onto the grid, then use the GBM transition probabilities?
    # Since we don't have the exact transition matrix for an arbitrary grid, we can use the paths.
    
    # Let's use a simpler approximation for the continuation value:
    # Continuation Value at t for a given state S is approx by the average of V_{t+1}(S_{t+1}) 
    # for paths that were at S at time t.
    
    # To make this efficient and vectorized:
    # 1. Interpolate V_{t+1} to the actual path values at t+1.
    # 2. For each path, we have S_t and V_{t+1}^{path}.
    # 3. We want to estimate E[V_{t+1} | S_t = g].
    #    We can bin the paths by S_t into grid cells, and average the V_{t+1} values in each cell.
    
    # Let's refine the grid to be based on the union of all time steps' quantiles to ensure good coverage.
    # Or just use the final step quantiles as they span the widest range.
    
    # Let's try a different strategy: 
    # Use the training paths to directly compute the optimal policy via a "pathwise" backward pass?
    # No, we need a policy function exercise(S, t).
    
    # Let's stick to the grid approach with conditional averaging.
    
    # 1. Prepare grid
    # 2. V_next = Payoff at last date (interpolated to grid? No, V_next is defined on grid)
    #    Actually, V is defined on the grid.
    
    V_grid = np.zeros_like(grid)
    
    # Last exercise date index in exercise_data is -1
    # Payoff at last date
    V_grid = np.maximum(K - grid, 0.0)
    
    # Discount factor per step
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # Backward loop
    # We iterate t from n_dates-1 down to 0 (indices in exercise_data)
    # exercise_data[:, t] is S_t
    # exercise_data[:, t+1] is S_{t+1}
    
    # Pre-compute indices for binning S_t into grid cells
    # We can use np.searchsorted to find the grid interval for each path's S_t
    
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        
        # 1. Interpolate V_next (which is currently V_grid for t+1) to the actual S_t_next values
        # V_next_values_on_paths = np.interp(S_t_next, grid, V_grid)
        
        # 2. Estimate Continuation Value on the grid
        # For each grid point g, CV(g) = E[V_next(S_{t+1}) | S_t = g]
        # Approximate by averaging V_next(S_{t+1}) for paths where S_t is in the bin around g.
        
        # Find bin indices for S_t
        # We can use the grid points as centers and assign each path to the nearest grid point?
        # Or use searchsorted to find intervals.
        
        # Let's use a simple binning:
        # Assign each path to the nearest grid point.
        # This is O(N * G) if naive, but we can vectorize.
        
        # Actually, a more accurate method is to use local regression or just nearest neighbor for simplicity and speed.
        # Given 4096 paths and 200 grid points, 4096 * 200 = 800k operations, which is fast.
        
        # Let's compute the nearest grid index for each path at time t.
        # dist = abs(S_t[:, None] - grid[None, :])
        # nearest_idx = np.argmin(dist, axis=1)
        
        # This creates a 4096 x 200 array, which is fine.
        
        # However, we need the average of V_next(S_{t+1}) for each bin.
        # V_next_on_paths = np.interp(S_t_next, grid, V_grid)
        
        # We can use np.bincount or similar.
        
        # Let's try a different, faster approximation:
        # Continuation value is approximately V_grid * discount? No.
        # For a put, if S is low, continuation is high.
        
        # Let's use the "Longstaff-Schwartz" style but with a fixed grid.
        # Actually, for a simple GBM, we can use the analytical solution for the continuation value
        # of a European put if we assume no exercise at t+1? No, that's not right.
        
        # Let's go back to the conditional expectation via binning.
        
        # Step 1: Interpolate V_grid to S_t_next
        V_next_on_paths = np.interp(S_t_next, grid, V_grid)
        
        # Step 2: Bin S_t into grid cells
        # We'll use the grid points as bin centers.
        # Find the closest grid point for each S_t.
        # To speed up, we can use searchsorted on the midpoints.
        
        # Midpoints between grid points
        if len(grid) > 1:
            midpoints = (grid[:-1] + grid[1:]) / 2.0
            # Find which bin each S_t falls into
            # Bin i corresponds to grid[i]
            # If S_t < mid[0], bin 0. If S_t > mid[-1], bin last.
            bin_indices = np.searchsorted(midpoints, S_t)
        else:
            bin_indices = np.zeros(len(S_t), dtype=int)
            
        # Step 3: Average V_next_on_paths for each bin
        # Sum and count
        total_val = np.bincount(bin_indices, weights=V_next_on_paths, minlength=len(grid))
        counts = np.bincount(bin_indices, minlength=len(grid))
        
        # Avoid division by zero
        counts_safe = np.where(counts > 0, counts, 1.0)
        cont_val = total_val / counts_safe
        
        # For bins with no paths, we can interpolate from neighbors or set to 0.
        # Interpolation is better.
        mask_empty = counts == 0
        if np.any(mask_empty):
            # Interpolate cont_val for empty bins
            # Use linear interpolation of the non-empty bins
            non_empty_idx = np.where(~mask_empty)[0]
            if len(non_empty_idx) > 0:
                # Create a full array with NaNs for empty
                cont_val_full = np.full_like(cont_val, np.nan)
                cont_val_full[~mask_empty] = cont_val[~mask_empty]
                # Interpolate
                x = np.arange(len(grid))
                y = cont_val_full
                # Fill NaNs
                mask_nan = np.isnan(y)
                if np.any(mask_nan):
                    y[mask_nan] = np.interp(x[mask_nan], x[~mask_nan], y[~mask_nan])
                    cont_val = y
            else:
                cont_val = np.zeros_like(grid)
                
        # Step 4: Discount continuation value
        cont_val_discounted = cont_val * discount
        
        # Step 5: Intrinsic value on grid
        intrinsic = np.maximum(K - grid, 0.0)
        
        # Step 6: Value is max(Intrinsic, Continuation)
        V_grid = np.maximum(intrinsic, cont_val_discounted)
        
        # Check time
        if time.time() > deadline:
            break
            
    # Now we have V_grid for t=0 (first exercise date in our data slice).
    # Wait, the loop went from n_dates-2 down to 0.
    # So V_grid is now the value at t=0 (the first exercise date).
    
    # We need to return an object with exercise(S, t) method.
    # The policy is: Exercise if Intrinsic(S, t) >= Continuation(S, t).
    # Which is equivalent to Intrinsic(S, t) >= V_grid[S] (since V = max(Intr, Cont)).
    # So, Exercise if Intrinsic(S) >= V_grid[S].
    
    # We need to store the V_grid for each time step to evaluate the policy at any t.
    # My loop overwrote V_grid. I need to store the decision boundary or the V_grid for each t.
    
    # Let's restart the backward induction but store the "Exercise" mask for each time step on the grid.
    
    # Re-initialize
    V_grid = np.zeros_like(grid)
    V_grid = np.maximum(K - grid, 0.0) # Last date
    
    # Store exercise masks
    exercise_masks = [None] * n_dates
    
    # Last date: Always exercise if intrinsic > 0? 
    # Actually, at the last date, you compare Intrinsic vs 0 (since it expires).
    # So Exercise if Intrinsic > 0.
    exercise_masks[-1] = (K - grid) > 0
    
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        
        V_next_on_paths = np.interp(S_t_next, grid, V_grid)
        
        if len(grid) > 1:
            midpoints = (grid[:-1] + grid[1:]) / 2.0
            bin_indices = np.searchsorted(midpoints, S_t)
        else:
            bin_indices = np.zeros(len(S_t), dtype=int)
            
        total_val = np.bincount(bin_indices, weights=V_next_on_paths, minlength=len(grid))
        counts = np.bincount(bin_indices, minlength=len(grid))
        counts_safe = np.where(counts > 0, counts, 1.0)
        cont_val = total_val / counts_safe
        
        mask_empty = counts == 0
        if np.any(mask_empty):
            non_empty_idx = np.where(~mask_empty)[0]
            if len(non_empty_idx) > 0:
                cont_val_full = np.full_like(cont_val, np.nan)
                cont_val_full[~mask_empty] = cont_val[~mask_empty]
                x = np.arange(len(grid))
                y = cont_val_full
                mask_nan = np.isnan(y)
                if np.any(mask_nan):
                    y[mask_nan] = np.interp(x[mask_nan], x[~mask_nan], y[~mask_nan])
                    cont_val = y
            else:
                cont_val = np.zeros_like(grid)
                
        cont_val_discounted = cont_val * discount
        intrinsic = np.maximum(K - grid, 0.0)
        
        V_grid = np.maximum(intrinsic, cont_val_discounted)
        
        # Exercise if Intrinsic >= Continuation
        # i.e., Intrinsic >= V_grid
        # Note: If Intrinsic == V_grid, it's indifferent. Usually exercise to get cash.
        exercise_masks[t] = intrinsic >= V_grid
        
        if time.time() > deadline:
            # If we run out of time, we have to stop.
            # We have V_grid for current t.
            # We can fill the rest with "Exercise if Intrinsic > 0" (conservative) or break.
            # Let's break and handle missing masks in the class.
            break
            
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, grid, masks, K):
            self.grid = grid
            self.masks = masks
            self.K = K
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            Returns: boolean vector
            """
            if time >= len(self.masks) or self.masks[time] is None:
                # Fallback: Exercise if Intrinsic > 0
                return (self.K - current_state) > 0
                
            # Interpolate the decision boundary? 
            # No, we have a mask for each grid point.
            # We need to map current_state to the grid and use the mask.
            # If a state is between grid points, we can interpolate the V_grid?
            # But we only stored the mask.
            # Better to store V_grid for each time step and compare Intrinsic vs V_grid.
            # Let's modify the loop to store V_grid instead of masks.
            
            # Since I already wrote the loop with masks, let's stick to it but handle interpolation.
            # Actually, the mask is binary. Interpolating a binary mask is bad.
            # We should have stored V_grid.
            
            # Let's refactor the return to use V_grids.
            
            pass # Placeholder
            
    # Refactor: Store V_grids instead of masks.
    V_grids = [None] * n_dates
    V_grid = np.maximum(K - grid, 0.0)
    V_grids[-1] = V_grid.copy()
    
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        
        V_next_on_paths = np.interp(S_t_next, grid, V_grid)
        
        if len(grid) > 1:
            midpoints = (grid[:-1] + grid[1:]) / 2.0
            bin_indices = np.searchsorted(midpoints, S_t)
        else:
            bin_indices = np.zeros(len(S_t), dtype=int)
            
        total_val = np.bincount(bin_indices, weights=V_next_on_paths, minlength=len(grid))
        counts = np.bincount(bin_indices, minlength=len(grid))
        counts_safe = np.where(counts > 0, counts, 1.0)
        cont_val = total_val / counts_safe
        
        mask_empty = counts == 0
        if np.any(mask_empty):
            non_empty_idx = np.where(~mask_empty)[0]
            if len(non_empty_idx) > 0:
                cont_val_full = np.full_like(cont_val, np.nan)
                cont_val_full[~mask_empty] = cont_val[~mask_empty]
                x = np.arange(len(grid))
                y = cont_val_full
                mask_nan = np.isnan(y)
                if np.any(mask_nan):
                    y[mask_nan] = np.interp(x[mask_nan], x[~mask_nan], y[~mask_nan])
                    cont_val = y
            else:
                cont_val = np.zeros_like(grid)
                
        cont_val_discounted = cont_val * discount
        intrinsic = np.maximum(K - grid, 0.0)
        
        V_grid = np.maximum(intrinsic, cont_val_discounted)
        V_grids[t] = V_grid.copy()
        
        if time.time() > deadline:
            break
            
    # Fill any remaining None V_grids with the last computed one or intrinsic
    for t in range(n_dates):
        if V_grids[t] is None:
             V_grids[t] = np.maximum(K - grid, 0.0)
             
    class ExercisePolicy:
        def __init__(self, grid, V_grids, K):
            self.grid = grid
            self.V_grids = V_grids
            self.K = K
            
        def exercise(self, current_state, time):
            if time >= len(self.V_grids):
                return (self.K - current_state) > 0
                
            V_t = self.V_grids[time]
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # Interpolate V_t to current_state
            V_current = np.interp(current_state, self.grid, V_t)
            
            # Exercise if Intrinsic >= V_current
            return intrinsic >= V_current
            
    return ExercisePolicy(grid, V_grids, K)

# EVOLVE-BLOCK-END
