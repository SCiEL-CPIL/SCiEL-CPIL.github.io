"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with 
    piecewise-linear value function approximation via quantile binning.
    
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
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
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
    if total_time_steps == n_steps + 1:
        exercise_indices = list(range(1, n_steps + 1))
    else:
        exercise_indices = list(range(total_time_steps - n_steps, total_time_steps))
    
    n_exercise_dates = len(exercise_indices)
    
    # Use adaptive binning based on the distribution of states
    # We'll use quantile-based bins for better resolution where needed
    
    # Collect all states at exercise dates to determine bin edges
    all_states = np.concatenate([training_paths[:, idx] for idx in exercise_indices])
    
    # Use 100 bins for good resolution
    n_bins = 100
    
    # Create bin edges using quantiles for better coverage
    # Extend range slightly to cover extreme values
    q_min = np.percentile(all_states, 0.5)
    q_max = np.percentile(all_states, 99.5)
    margin = 0.1 * (q_max - q_min) if q_max > q_min else 1.0
    s_min = max(0.1, q_min - margin)
    s_max = q_max + margin
    
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
            for k in range(j, -1, -1):
                if all_V_vals[k] is None:
                    all_V_vals[k] = payoff(bin_centers)
            break
            
        current_col = exercise_indices[j]
        next_col = exercise_indices[j + 1]
        
        S_current = training_paths[:, current_col]
        S_next = training_paths[:, next_col]
        
        # Evaluate V_next at S_next for all paths
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        
        # Continuation value for each path
        cont_vals = discount * V_next_at_paths
        
        # For each bin of S_current, compute the average continuation value
        V_cont_at_bins = np.zeros(n_bins)
        
        # Use np.digitize to assign paths to bins efficiently
        bin_indices = np.digitize(S_current, bin_edges, right=False)
        bin_indices = np.clip(bin_indices, 1, n_bins) - 1  # 0-indexed
        
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
            # Interpolate empty bins
            for b in range(n_bins):
                if not valid[b]:
                    # Find nearest valid neighbors
                    left = b - 1
                    while left >= 0 and not valid[left]:
                        left -= 1
                    right = b + 1
                    while right < n_bins and not valid[right]:
                        right += 1
                    
                    if left >= 0 and right < n_bins:
                        # Linear interpolation
                        frac = (b - left) / (right - left)
                        V_cont_at_bins[b] = (1 - frac) * V_cont_at_bins[left] + frac * V_cont_at_bins[right]
                    elif left >= 0:
                        V_cont_at_bins[b] = V_cont_at_bins[left]
                    elif right < n_bins:
                        V_cont_at_bins[b] = V_cont_at_bins[right]
                    else:
                        V_cont_at_bins[b] = payoff(bin_centers[b])
        
        # The value function at this date: V = max(payoff, continuation)
        payoff_at_bins = payoff(bin_centers)
        V_vals = np.maximum(payoff_at_bins, V_cont_at_bins)
        
        all_V_vals[j] = V_vals
    
    # Now determine the exercise policy for each date
    # For each date, we need to determine whether to exercise given S
    # Exercise if payoff(S) >= continuation_value(S)
    # i.e., payoff(S) >= discount * E[V_{t+1}(S_{t+1}) | S_t = S]
    
    # We'll store the continuation values at bin centers for each date
    # and use interpolation at evaluation time
    
    # Store continuation values for each date (excluding last, which is just payoff)
    cont_vals_at_bins = [None] * n_exercise_dates
    cont_vals_at_bins[-1] = np.zeros(n_bins)  # At maturity, no continuation
    
    for j in range(n_exercise_dates - 1):
        if time.time() > deadline - 2.0:
            # Use simple approximation
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
        bin_indices = np.clip(bin_indices, 1, n_bins) - 1
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, cont_vals)
        np.add.at(bin_counts, bin_indices, 1.0)
        
        valid = bin_counts > 0
        cont_at_bins = np.zeros(n_bins)
        cont_at_bins[valid] = bin_sums[valid] / bin_counts[valid]
        
        # Fill empty bins
        if not np.all(valid):
            for b in range(n_bins):
                if not valid[b]:
                    left = b - 1
                    while left >= 0 and not valid[left]:
                        left -= 1
                    right = b + 1
                    while right < n_bins and not valid[right]:
                        right += 1
                    
                    if left >= 0 and right < n_bins:
                        frac = (b - left) / (right - left)
                        cont_at_bins[b] = (1 - frac) * cont_at_bins[left] + frac * cont_at_bins[right]
                    elif left >= 0:
                        cont_at_bins[b] = cont_at_bins[left]
                    elif right < n_bins:
                        cont_at_bins[b] = cont_at_bins[right]
                    else:
                        cont_at_bins[b] = payoff(bin_centers[b])
        
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
