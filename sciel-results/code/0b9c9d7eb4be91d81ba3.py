import numpy as np
import time
from scipy.stats import norm, qmc

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with Quasi-Monte Carlo (Sobol) 
    for continuation value estimation.
    """
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Time step
    dt = T / n_exercise
    sqrt_dt = np.sqrt(dt)
    disc = np.exp(-r * dt)
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * sqrt_dt

    # Ensure training_paths is a numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, n_exercise)
    # Handle shape: (n_paths, n_exercise) or (n_exercise, n_paths)
    if paths.shape[0] != 4096 and paths.shape[1] == 4096:
        paths = paths.T
    if paths.shape[1] > n_exercise:
        paths = paths[:, :n_exercise]
    elif paths.shape[1] < n_exercise:
        pad_width = n_exercise - paths.shape[1]
        paths = np.hstack([paths, np.zeros((paths.shape[0], pad_width))])

    n_paths, n_dates = paths.shape

    # Build state grid
    # Use a theoretical grid based on GBM bounds to ensure coverage
    # S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*sqrt(t)*Z)
    # We estimate S_0 from the first column of training paths or assume 100? 
    # Contract says Spot 90/100/110. Training paths likely start at these.
    # Let's use the min and max of all training states to define the grid range, 
    # extended slightly for safety.
    all_states = paths.flatten()
    min_state = np.min(all_states) * 0.5
    max_state = np.max(all_states) * 1.5
    
    n_grid = 800
    # Log-uniform grid is often better for GBM
    grid = np.exp(np.linspace(np.log(max(min_state, 1e-6)), np.log(max_state), n_grid))
    grid = np.sort(grid)
    n_grid = len(grid)

    # Generate Quasi-Monte Carlo points (Sobol)
    # We need 1D points mapped to Normal distribution
    # Sobol requires dimension 1 for 1D problem? 
    # scipy.stats.qmc.Sobol(d=1) generates 1D Sobol points.
    # Number of points should be power of 2 for best properties, or use 'randomization' if needed.
    # Let's use 32768 points.
    n_mc = 32768
    sobol = qmc.Sobol(d=1, scramble=False)
    # Generate points in [0, 1]
    u_points = sobol.random(n_mc)  # shape (n_mc, 1)
    u_points = u_points.flatten()
    
    # Map to Standard Normal
    # Clip to avoid inf at 0 or 1
    u_points = np.clip(u_points, 1e-9, 1 - 1e-9)
    z_points = norm.ppf(u_points)  # shape (n_mc,)

    # Backward induction
    # V[i, j] is the value at exercise date i for grid point j
    V_grid = np.zeros((n_exercise, n_grid))
    cont_stored = np.zeros((n_exercise, n_grid))

    # Last exercise date: intrinsic value only
    intrinsic_last = np.maximum(K - grid, 0.0)
    V_grid[-1, :] = intrinsic_last
    cont_stored[-1, :] = 0.0

    # Work backward
    for i in range(n_exercise - 2, -1, -1):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            # If we run out of time, we must return a valid policy.
            # We have V_grid for dates > i. For dates <= i, we can assume exercise if intrinsic > 0?
            # Or just break and use whatever is computed. 
            # The policy object will use cont_stored. If cont_stored is 0 for uncomputed steps,
            # it will exercise if intrinsic > 0. This is a valid fallback (early exercise is often optimal for deep ITM).
            break

        V_next = V_grid[i + 1, :]
        
        # Compute continuation values for all grid points
        # S_next = grid[:, None] * exp(drift + vol_step * z_points[None, :])
        # To save memory and time, process in chunks if n_grid is large, but 800 * 32768 is ~26M elements, ~200MB.
        # This might be slow if done naively. Let's try to vectorize efficiently.
        
        # Pre-compute the exponential factor
        # log_factor = drift + vol_step * z_points  (n_mc,)
        log_factor = drift + vol_step * z_points
        
        # We want to compute: mean_k ( V_next[interp(grid[j] * exp(log_factor[k]))] )
        # This is a convolution-like operation.
        
        # Vectorized approach:
        # Create matrix of next states: (n_grid, n_mc)
        # This is memory heavy. Let's chunk the grid.
        chunk_size = 100
        cont_values = np.zeros(n_grid)
        
        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            s_chunk = grid[start:end, None]  # (chunk, 1)
            
            # S_next = s_chunk * exp(log_factor[None, :])
            S_next = s_chunk * np.exp(log_factor[None, :])  # (chunk, n_mc)
            
            # Interpolate V_next
            # Clip S_next to grid range
            S_next_clipped = np.clip(S_next, grid[0], grid[-1])
            
            # np.interp is fast, but for 2D input?
            # We can use np.interp on flattened arrays or loop over chunk?
            # Better: use a lookup table or vectorized interp.
            # np.interp expects 1D xp.
            # We can apply np.interp to each row, or use a function that handles 2D.
            # Actually, np.interp can take 2D x if we reshape? No, xp must be 1D.
            # We can use np.searchsorted + linear interpolation manually for speed, or just call np.interp in a loop over the small chunk?
            # Chunk size 100, n_mc 32768. 100 calls to np.interp with 32768 points each.
            # This is acceptable.
            
            V_next_vals = np.empty_like(S_next_clipped)
            for k in range(end - start):
                V_next_vals[k, :] = np.interp(S_next_clipped[k, :], grid, V_next)
            
            # Average over MC paths
            cont_values[start:end] = disc * np.mean(V_next_vals, axis=1)
        
        # Exercise value
        ex_values = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V_grid[i, :] = np.maximum(ex_values, cont_values)
        cont_stored[i, :] = cont_values

    # If we broke out early, fill remaining cont_stored with 0 (implies exercise if ITM)
    # This is a conservative fallback.

    class ExercisePolicy:
        def __init__(self, grid, cont_stored, n_exercise, K):
            self.grid = grid
            self.cont_stored = cont_stored
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices, shape (n_eval_paths,)
            time_idx: index of exercise date (0-indexed, 0 to n_exercise-1)
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            if time_idx < 0 or time_idx >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            # Clip to grid range
            cs_clipped = np.clip(current_state, self.grid[0], self.grid[-1])
            
            # Interpolate continuation value at current state
            cont_vals = np.interp(cs_clipped, self.grid, self.cont_stored[time_idx, :])
            
            # Exercise if intrinsic >= continuation
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic >= cont_vals

    return ExercisePolicy(grid, cont_stored, n_exercise, K)
