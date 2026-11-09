"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with 
    piecewise-linear value function approximation via high-resolution binning.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths.
    contract : dict
        Contract parameters.
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Convert training_paths to numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, total_time_steps = training_paths.shape
    
    # Determine exercise dates: 12 monthly exercise dates, no time-0 exercise
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Payoff function for put: max(K - S, 0)
    def payoff(S):
        return np.maximum(K - S, 0.0)
    
    # Determine exercise indices
    # The problem states "12 monthly exercise dates (no time-0 exercise)".
    # Training paths likely have n_steps + 1 time points (0 to 12).
    # Exercise is allowed at indices 1, 2, ..., 12.
    if total_time_steps == n_steps + 1:
        exercise_indices = list(range(1, n_steps + 1))
    else:
        # Fallback: assume last n_steps columns are the exercise dates
        exercise_indices = list(range(total_time_steps - n_steps, total_time_steps))
    
    n_exercise_dates = len(exercise_indices)
    
    # Determine bin edges
    # We want to cover the range of spot prices observed in the training data.
    # For a put, the value is 0 for high S. The critical region is S < K.
    # Let's use a range from 0 to max_observed * 1.1 to be safe.
    all_states = np.concatenate([training_paths[:, idx] for idx in exercise_indices])
    s_max = np.max(all_states) * 1.05
    s_min = 0.0
    
    # Use 500 bins for higher resolution
    n_bins = 500
    bin_edges = np.linspace(s_min, s_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Value function at each exercise date, stored as values at bin_centers
    all_V_vals = [None] * n_exercise_dates
    
    # At the last exercise date, V = payoff
    all_V_vals[-1] = payoff(bin_centers)
    
    # Work backwards from second-to-last to first exercise date
    for j in range(n_exercise_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, use simple approximation for remaining dates
            # Use the current best guess (payoff or previous V)
            for k in range(j, -1, -1):
                if all_V_vals[k] is None:
                    all_V_vals[k] = payoff(bin_centers)
            break
            
        current_col = exercise_indices[j]
        next_col = exercise_indices[j + 1]
        
        S_current = training_paths[:, current_col]
        S_next = training_paths[:, next_col]
        
        # Evaluate V_next at S_next for all paths
        # np.interp is fast and handles out-of-bounds by clipping
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        
        # Continuation value for each path
        cont_vals = discount * V_next_at_paths
        
        # For each bin of S_current, compute the average continuation value
        V_cont_at_bins = np.zeros(n_bins)
        
        # Assign paths to bins
        # np.digitize returns indices 1..n_bins for values in [edge[i-1], edge[i])
        # We want 0-indexed bins 0..n_bins-1
        # Values > s_max will get n_bins, values < s_min will get 0
        bin_indices = np.digitize(S_current, bin_edges, right=False)
        # Clip to valid range [0, n_bins-1]
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute sum and count for each bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, cont_vals)
        np.add.at(bin_counts, bin_indices, 1.0)
        
        # Compute averages, handling empty bins
        valid = bin_counts > 0
        V_cont_at_bins[valid] = bin_sums[valid] / bin_counts[valid]
        
        # Fill empty bins by interpolation from neighbors
        if not np.all(valid):
            # Find indices of valid bins
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                # Interpolate for empty bins
                # We can use np.interp on the bin centers
                empty_indices = np.where(~valid)[0]
                V_cont_at_bins[empty_indices] = np.interp(
                    bin_centers[empty_indices], 
                    bin_centers[valid_indices], 
                    V_cont_at_bins[valid_indices]
                )
            else:
                # All bins empty? Unlikely. Fallback to payoff.
                V_cont_at_bins[:] = payoff(bin_centers)
        
        # The value function at this date: V = max(payoff, continuation)
        payoff_at_bins = payoff(bin_centers)
        V_vals = np.maximum(payoff_at_bins, V_cont_at_bins)
        
        all_V_vals[j] = V_vals
    
    # Now determine the exercise policy for each date
    # We need to store the continuation values for each date to make the decision
    # Exercise if payoff(S) >= continuation_value(S)
    
    cont_vals_at_bins = [None] * n_exercise_dates
    # At the last date, continuation is 0 (or effectively, we always exercise if in the money, 
    # but the logic payoff >= 0 handles it. However, for the policy object, 
    # we can just set cont to 0 so payoff >= 0 is true if ITM).
    cont_vals_at_bins[-1] = np.zeros(n_bins)
    
    for j in range(n_exercise_dates - 1):
        if time.time() > deadline - 2.0:
            # Use simple approximation: assume continuation is slightly less than payoff
            # or just use the payoff itself to force exercise if ITM?
            # Better to use the last calculated V if available, or just payoff.
            # If we break early, we might have incomplete V_vals.
            # Let's just set continuation to payoff * 0.9 to encourage exercise if close.
            cont_vals_at_bins[j] = payoff(bin_centers) * 0.9
            continue
        
        current_col = exercise_indices[j]
        next_col = exercise_indices[j + 1]
        
        S_current = training_paths[:, current_col]
        S_next = training_paths[:, next_col]
        
        # Evaluate V_next at S_next
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        cont_vals = discount * V_next_at_paths
        
        # Average by bin
        bin_indices = np.digitize(S_current, bin_edges, right=False)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, cont_vals)
        np.add.at(bin_counts, bin_indices, 1.0)
        
        valid = bin_counts > 0
        cont_at_bins = np.zeros(n_bins)
        cont_at_bins[valid] = bin_sums[valid] / bin_counts[valid]
        
        # Fill empty bins
        if not np.all(valid):
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                empty_indices = np.where(~valid)[0]
                cont_at_bins[empty_indices] = np.interp(
                    bin_centers[empty_indices], 
                    bin_centers[valid_indices], 
                    cont_at_bins[valid_indices]
                )
            else:
                cont_at_bins[:] = payoff(bin_centers)
        
        cont_vals_at_bins[j] = cont_at_bins
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, cont_vals, bin_edges, bin_centers, K, r, sigma, T, n_steps):
            self.cont_vals = cont_vals
            self.bin_edges = bin_edges
            self.bin_centers = bin_centers
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each path.
            
            Parameters:
            -----------
            current_state : array-like, shape (n_paths,)
                Current spot prices.
            time : float or int
                Current time (or step index).
                
            Returns:
            --------
            boolean array of shape (n_paths,)
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # Determine which exercise date we're at
            dt = self.T / self.n_steps
            
            if isinstance(time, (int, np.integer)):
                step = int(time)
            else:
                step = int(round(time / dt))
            
            # Clamp step to valid range
            # Exercise dates are 1..n_steps
            step = max(1, min(step, self.n_steps))
            
            # Get the continuation values for this exercise date
            # step corresponds to exercise date index step - 1
            date_idx = step - 1
            date_idx = max(0, min(date_idx, len(self.cont_vals) - 1))
            
            if date_idx >= len(self.cont_vals):
                # At or after maturity, exercise if in the money
                return current_state < self.K
            
            # Get continuation values at bin centers
            cont_at_bins = self.cont_vals[date_idx]
            
            # Interpolate continuation value at current states
            cont_vals = np.interp(current_state, self.bin_centers, cont_at_bins)
            
            # Payoff at current states
            payoffs = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if payoff >= continuation value
            return payoffs >= cont_vals
    
    return ExercisePolicy(cont_vals_at_bins, bin_edges, bin_centers, K, r, sigma, T, n_steps)
# EVOLVE-BLOCK-END
