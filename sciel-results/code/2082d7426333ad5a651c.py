# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using backward induction on a high-resolution 
    quantile grid with piecewise linear interpolation.
    
    This approach constructs a detailed value function at each time step by 
    averaging the continuation values on a fine grid of state quantiles derived 
    from the training paths. This avoids the global approximation errors of 
    polynomial fits, particularly near the exercise boundary where the value 
    function is non-smooth (kinked).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_cols)
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise data columns
    # Task: "12 monthly exercise dates (no time-0 exercise)"
    # If n_cols > n_dates, assume first col is t0 (S0) and rest are exercise dates.
    # If n_cols == n_dates, assume all cols are exercise dates.
    if n_cols > n_dates:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    # Ensure we have exactly n_dates columns
    if exercise_data.shape[1] < n_dates:
        # Fallback: if data is shorter, we can't do much. 
        # But standard contract should provide correct shape.
        pass

    n_paths_eff, n_steps_eff = exercise_data.shape
    if n_steps_eff != n_dates:
        n_dates = n_steps_eff

    # Define grid size. 
    # We want a fine grid to capture the kink. 
    # 1000-2000 points is usually sufficient for this precision.
    # Let's use 1000 quantiles per step.
    grid_size = 1000
    
    # Pre-compute discount factor
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # V_grids[t] will store the value function at time t on the grid defined by S_grids[t]
    # S_grids[t] are the quantiles of S_t from the training paths
    # V_grids[t] are the corresponding optimal values
    
    S_grids = [None] * n_dates
    V_grids = [None] * n_dates
    
    # Initialize at T (last date, index n_dates - 1)
    S_T = exercise_data[:, -1]
    # Sort S_T to create the grid
    sorted_indices_T = np.argsort(S_T)
    S_grid_T = S_T[sorted_indices_T]
    # Quantile grid: take evenly spaced indices
    # We want grid_size points. 
    # Use np.linspace to pick indices
    idx_grid_T = np.linspace(0, len(S_T) - 1, grid_size, dtype=int)
    S_grid_T = S_T[idx_grid_T]
    # Ensure uniqueness? np.interp requires increasing x. 
    # If there are duplicates, we might have issues. 
    # Quantiles from random paths are unlikely to be identical, but possible.
    # Let's add a tiny epsilon to ensure strict increase if needed, or just use np.unique?
    # Actually, np.interp handles non-decreasing x, but strict increase is safer.
    # Let's just use the sorted unique values? No, that changes the distribution.
    # Better: just use the selected quantiles. If duplicates exist, np.interp works 
    # as long as x is non-decreasing.
    
    S_grids[-1] = S_grid_T
    V_grids[-1] = np.maximum(K - S_grid_T, 0.0)
    
    # Backward loop
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        
        # 1. Evaluate V_{t+1}(S_{t+1}) using the grid from t+1
        # V_{t+1} is defined on S_grids[t+1]
        # We need V_{t+1}(S_t_next) for each path
        V_next_on_paths = np.interp(S_t_next, S_grids[t+1], V_grids[t+1])
        
        # 2. Compute Continuation Value C_t(S_t)
        # C_t(S_t) is the expected discounted value of V_{t+1}(S_{t+1}) given S_t.
        # Since we don't have an analytical expectation for the interpolated function, 
        # we use the training paths to estimate the conditional expectation.
        # This is the standard "Monte Carlo" step in LSM or similar methods, 
        # but here we are aggregating to a grid.
        
        # Strategy: 
        # For each path i, we have S_t[i] and V_{t+1}(S_{t+1}[i]).
        # We want to estimate E[discount * V_{t+1}(S_{t+1}) | S_t = s].
        # We can do this by binning S_t into the same quantile grid used for S_t.
        
        # Create grid for S_t
        sorted_indices_t = np.argsort(S_t)
        S_sorted_t = S_t[sorted_indices_t]
        idx_grid_t = np.linspace(0, len(S_t) - 1, grid_size, dtype=int)
        S_grid_t = S_sorted_t[idx_grid_t]
        S_grids[t] = S_grid_t
        
        # To compute C_t on this grid, we can use a local averaging approach 
        # or regression. 
        # Given the computational budget, a simple and robust method is 
        # to use the paths directly to estimate the conditional expectation 
        # via a nearest-neighbor or binning approach on the sorted S_t.
        
        # Let's use a "binned average" approach which is fast and stable.
        # We define bins based on the quantiles of S_t.
        # For each bin (corresponding to a grid point), we average the 
        # discounted next-step values of the paths falling in that bin.
        
        # However, simply binning by S_t quantiles might be noisy if bins are small.
        # With 4096 paths and 1000 bins, each bin has ~4 paths. This is very noisy.
        # We need a smoother estimate.
        
        # Alternative: Use a polynomial or spline fit to the scatter plot of 
        # (S_t, discount * V_{t+1}(S_{t+1}))?
        # Or, use the fact that S_{t+1} is related to S_t.
        
        # Let's try a different approach for the continuation value estimation:
        # Since we have a grid for S_t, we can compute the continuation value 
        # on this grid by averaging the discounted future values for paths 
        # close to each grid point.
        
        # To improve efficiency and accuracy, we can use a "moving average" 
        # or "kernel density estimation" style approach, but that's O(N^2) or slow.
        
        # Let's stick to a simpler, faster method: 
        # Fit a low-order polynomial to (S_t, discounted_V_next) and evaluate it on the grid?
        # The previous program failed because it fit the *total* value V_t with a polynomial, 
        # which is kinked. But the *continuation* value C_t(S_t) is smooth!
        # C_t(S) = E[discount * V_{t+1}(S_{t+1}) | S_t=S].
        # If V_{t+1} is piecewise linear, C_t is a smooth function (integral of a step function).
        # So, fitting C_t with a polynomial or spline is much more appropriate than fitting V_t.
        
        # Let's fit a cubic polynomial (or higher) to the continuation values on the paths.
        # Then evaluate this polynomial on the S_grid_t to get C_t(S_grid_t).
        
        y_cont = discount * V_next_on_paths
        x_cont = S_t
        
        # Fit polynomial. Order 4 or 5 should be enough for smoothness.
        # Use np.polyfit.
        try:
            # Regularize? np.polyfit can be ill-conditioned. 
            # Let's use np.linalg.lstsq with a Vandermonde matrix.
            order = 5
            X = np.vander(x_cont, N=order+1, increasing=True)
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y_cont, rcond=None)
            
            # Evaluate on grid
            X_grid = np.vander(S_grid_t, N=order+1, increasing=True)
            C_t_on_grid = X_grid @ coeffs
        except:
            # Fallback: simple average
            C_t_on_grid = np.full(grid_size, np.mean(y_cont))
            
        # 3. Compute Intrinsic Value on grid
        Intrinsic_t_on_grid = np.maximum(K - S_grid_t, 0.0)
        
        # 4. Compute V_t(S_t) = max(Intrinsic, Continuation)
        V_t_on_grid = np.maximum(Intrinsic_t_on_grid, C_t_on_grid)
        
        # Store for next iteration
        V_grids[t] = V_t_on_grid
        
        # Check time
        if time.time() > deadline:
            # If we run out of time, we have V_grids for t..T.
            # We need to fill the rest (0..t-1).
            # We can fill them with the last computed grid (constant in time approximation)
            # or intrinsic. Using the last computed grid is better.
            break
            
    # Fill any remaining None V_grids
    for t in range(n_dates):
        if V_grids[t] is None:
            # Find next non-None
            for j in range(t+1, n_dates):
                if V_grids[j] is not None:
                    V_grids[t] = V_grids[j]
                    S_grids[t] = S_grids[j]
                    break
            else:
                # All None? Fallback to intrinsic on a simple grid
                S_simple = np.linspace(0, 2*K, grid_size)
                V_simple = np.maximum(K - S_simple, 0.0)
                V_grids[t] = V_simple
                S_grids[t] = S_simple

    class ExercisePolicy:
        def __init__(self, S_grids_list, V_grids_list, K):
            self.S_grids = S_grids_list
            self.V_grids = V_grids_list
            self.K = K
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            Returns: boolean vector
            """
            if time >= len(self.S_grids):
                return (self.K - current_state) > 0
                
            S_grid = self.S_grids[time]
            V_grid = self.V_grids[time]
            
            # Interpolate the value function V_t(S) for the current states
            V_current = np.interp(current_state, S_grid, V_grid)
            
            # Intrinsic value
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Intrinsic >= V_current
            # Note: V_current is the estimated total value.
            # If Intrinsic >= V_current, it implies Intrinsic >= Continuation (since V = max(Intr, Cont)).
            # Actually, if V_current is the fitted max(Intr, Cont), then:
            # If we are in the exercise region, V_current should be close to Intrinsic.
            # If we are in the continuation region, V_current should be > Intrinsic.
            # So, exercise if Intrinsic >= V_current (with a small tolerance for numerical errors).
            
            return intrinsic >= V_current
            
    return ExercisePolicy(S_grids, V_grids, K)

# EVOLVE-BLOCK-END
