"""Rewritten program: model-and-optimize approach for Bermudan put exercise policy."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    
    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Extract parameters from contract
    spot0 = contract.get('spot', 100.0)
    K = contract.get('strike', 100.0)
    r = contract.get('rate', 0.05)
    sigma = contract.get('vol', 0.2)
    T = contract.get('maturity', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_dates)
    # n_dates is the number of exercise dates (12 monthly, no time-0)
    if training_paths is not None and len(training_paths.shape) == 2:
        n_train = training_paths.shape[0]
        n_dates_actual = training_paths.shape[1]
    else:
        n_train = 4096
        n_train = 4096
        n_dates_actual = n_dates
    
    # Time step
    dt = T / n_dates_actual
    disc = math.exp(-r * dt)
    
    # We'll use a threshold-based policy: exercise at date i if S < threshold_i
    # We need to find optimal thresholds for each of the n_dates exercise dates
    
    # Approach: Use the training paths to estimate optimal exercise thresholds
    # via backward induction on a discretized state space
    
    # First, let's estimate the continuation value using the training paths
    # We'll use a simple approach: for each exercise date, find the threshold
    # that maximizes the expected discounted payoff
    
    # Let's use a quantile-based threshold approach
    # For each exercise date i (from last to first), we can estimate the
    # optimal threshold by looking at the distribution of future values
    
    # Better approach: Use backward induction with empirical distributions
    # For each date i, given the current state S, the value is:
    # max(K - S, disc * E[continuation_value(S_{i+1}) | S_i = S])
    
    # Since we have training paths, we can use them to estimate conditional expectations
    
    # Let's use a grid-based approach
    # Discretize the state space
    s_min = 1.0
    s_max = max(spot0 * 3, K * 2)
    n_grid = 200
    s_grid = np.linspace(s_min, s_max, n_grid)
    
    # For each exercise date, we'll store the optimal threshold
    # and the value function on the grid
    
    # Initialize value at the last exercise date (n_dates - 1)
    # At the last date, it's optimal to exercise if K - S > 0
    # So threshold = K
    
    # Value function: V_i(s) = max(K - s, disc * E[V_{i+1}(S_{i+1}) | S_i = s])
    
    # We'll use the training paths to estimate the conditional expectations
    # For each grid point s, find paths where S_i is close to s, then
    # look at the distribution of S_{i+1}
    
    # This is computationally expensive, so let's use a simpler approach:
    # Use the analytical GBM transition to estimate E[V_{i+1}(S_{i+1}) | S_i = s]
    
    # Under risk-neutral GBM:
    # S_{i+1} = S_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # where Z ~ N(0,1)
    
    # So given S_i = s, S_{i+1} is lognormal with:
    # ln(S_{i+1}) ~ N(ln(s) + (r - sigma^2/2)*dt, sigma^2*dt)
    
    # We can use Monte Carlo to estimate E[V_{i+1}(S_{i+1}) | S_i = s]
    
    # For efficiency, let's use a fixed set of random numbers
    n_mc = 500
    rng = np.random.RandomState(42)
    z_mc = rng.randn(n_mc)
    
    # Backward induction
    # V[i] is the value function at date i, stored on the grid
    # We'll store thresholds as well
    
    V = np.zeros((n_dates, n_grid))
    thresholds = np.zeros(n_dates)
    
    # Last exercise date: always exercise if in-the-money
    V[-1] = np.maximum(K - s_grid, 0.0)
    thresholds[-1] = K  # Exercise if S < K
    
    # Work backwards
    for i in range(n_dates - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            break
        
        # For each grid point, compute continuation value
        # S_{i+1} = s * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        drift = (r - 0.5 * sigma * sigma) * dt
        vol_dt = sigma * math.sqrt(dt)
        
        # For each grid point s_grid[j], compute E[V[i+1](S_{i+1})]
        # S_{i+1} = s_grid[j] * exp(drift + vol_dt * Z)
        
        # Vectorized computation
        # s_grid has shape (n_grid,), z_mc has shape (n_mc,)
        # We want for each j: mean over m of V[i+1](s_grid[j] * exp(drift + vol_dt * z_mc[m]))
        
        # Compute log S_{i+1} for all combinations
        # log_S_next = log(s_grid[j]) + drift + vol_dt * z_mc[m]
        # S_next = exp(log_S_next)
        
        # To keep memory manageable, process in chunks
        chunk_size = 50
        cont_val = np.zeros(n_grid)
        
        for chunk_start in range(0, n_grid, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_grid)
            s_chunk = s_grid[chunk_start:chunk_end]  # (chunk_size,)
            
            # log_S_next: (chunk_size, n_mc)
            log_S_next = np.log(s_chunk)[:, None] + drift + vol_dt * z_mc[None, :]
            S_next = np.exp(log_S_next)  # (chunk_size, n_mc)
            
            # Interpolate V[i+1] at S_next
            # V[i+1] is defined on s_grid
            V_interp = np.interp(S_next, s_grid, V[i + 1], left=0.0, right=0.0)
            
            # Average over MC samples
            cont_val[chunk_start:chunk_end] = np.mean(V_interp, axis=1)
        
        # Total value: max(K - s, disc * cont_val)
        exercise_val = np.maximum(K - s_grid, 0.0)
        V[i] = np.maximum(exercise_val, disc * cont_val)
        
        # Find threshold: the largest s where exercise_val >= disc * cont_val
        # i.e., where K - s >= disc * cont_val
        # Exercise if K - s >= disc * cont_val
        # i.e., s <= K - disc * cont_val
        
        # Find the crossover point
        diff = exercise_val - disc * cont_val
        # Find where diff changes from positive to negative
        # diff is positive for small s, negative for large s
        
        # Find the largest index where diff >= 0
        positive_mask = diff >= 0
        if np.any(positive_mask):
            last_pos = np.where(positive_mask)[0][-1]
            # Interpolate to find the exact threshold
            if last_pos < n_grid - 1:
                # Linear interpolation between last_pos and last_pos+1
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d2 != d1:
                    frac = d1 / (d1 - d2)
                    thresholds[i] = s_grid[last_pos] + frac * (s_grid[last_pos + 1] - s_grid[last_pos])
                else:
                    thresholds[i] = s_grid[last_pos]
            else:
                thresholds[i] = s_grid[last_pos]
        else:
            # Never exercise
            thresholds[i] = 0.0
    
    # Check if we completed backward induction
    if time.time() > deadline:
        # If we didn't complete, use simple thresholds
        for i in range(n_dates):
            if thresholds[i] == 0.0 and i < n_dates - 1:
                thresholds[i] = K
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, s_grid, V, n_dates, K):
            self.thresholds = thresholds
            self.s_grid = s_grid
            self.V = V
            self.n_dates = n_dates
            self.K = K
        
        def exercise(self, current_state, time_idx):
            """
            Determine whether to exercise at each path.
            
            Args:
                current_state: array of current spot prices
                time_idx: the current exercise date index (0 to n_dates-1)
            
            Returns:
                Boolean vector indicating whether to exercise
            """
            # time_idx is the index of the current exercise date
            # Exercise if S < threshold[time_idx]
            if time_idx < 0 or time_idx >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[time_idx]
            return current_state < threshold
    
    return ExercisePolicy(thresholds, s_grid, V, n_dates_actual, K)

# EVOLVE-BLOCK-END
