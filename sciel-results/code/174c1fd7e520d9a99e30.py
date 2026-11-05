"""Bermudan put exercise-policy research with reconstructed value function thresholds."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt

    def exercise(self, current_state, time_index):
        """
        current_state: array of stock prices
        time_index: integer index of the exercise date (0 to 11)
        Returns: boolean array
        """
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    
    # Determine shape of training_paths
    n_dates, n_paths = training_paths.shape
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # Initialize values
    V = np.zeros((n_dates, n_paths))
    
    # Terminal payoff
    V[-1] = np.maximum(K - training_paths[-1], 0.0)
    
    # Store interpolation tables for each time step to reconstruct thresholds later
    # Each entry will be (bin_centers, bin_values)
    interp_tables = [None] * n_dates
    
    # Parameters for smoothing/interpolation
    n_bins = 500  # High resolution for accurate reconstruction
    
    # Backward Induction
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 15.0:
            # Fallback: if running out of time, use a simple heuristic for remaining steps
            # For a put, the threshold increases as we get closer to maturity (approaching K)
            # but is generally below K. A rough approximation is K * (1 - 0.1 * time_to_maturity)
            for t_rem in range(t, -1, -1):
                time_to_maturity = (n_dates - 1 - t_rem) * dt
                # Simple heuristic: threshold is lower when more time remains
                # This is a rough guess to avoid crashing
                interp_tables[t_rem] = (np.array([K*0.5, K*0.9]), np.array([0.0, 0.0])) 
            break
        
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # 1. Sort S_t and V_next
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # 2. Create bin edges using quantiles for roughly equal population per bin
        # Quantile binning ensures we have data in the tails, which is crucial for the threshold
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(sorted_S, quantiles)
        # Ensure unique edges and handle duplicates
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            # Degenerate case
            cont_val = discount_factor * np.mean(V_next)
            cont_values = np.full(n_paths, cont_val)
            payoff_now = np.maximum(K - S_t, 0.0)
            V[t] = np.maximum(payoff_now, cont_values)
            # Store a flat table
            interp_tables[t] = (np.array([0.0, 1.0]), np.array([cont_val, cont_val]))
            continue
        
        # Assign bins
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # 3. Compute bin means
        n_actual_bins = len(bin_edges) - 1
        bin_sums = np.zeros(n_actual_bins)
        bin_counts = np.zeros(n_actual_bins)
        
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        np.add.at(bin_counts, bin_indices, 1)
        
        valid_bins = bin_counts > 0
        bin_means = np.zeros(n_actual_bins)
        bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
        
        # 4. Fill empty bins via interpolation
        valid_bin_indices = np.where(valid_bins)[0]
        if len(valid_bin_indices) > 0 and len(valid_bin_indices) < n_actual_bins:
            invalid_bins = np.where(~valid_bins)[0]
            bin_means[invalid_bins] = np.interp(invalid_bins, valid_bin_indices, bin_means[valid_bin_indices])
        elif len(valid_bin_indices) == 0:
            bin_means[:] = np.mean(V_next)
            
        # 5. Define bin centers for interpolation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Store the table for later threshold extraction
        interp_tables[t] = (bin_centers, bin_means)
        
        # 6. Interpolate back to original S_t to get continuation values for V[t]
        # We interpolate the *undiscounted* expected future value, then discount
        cont_values_sorted = np.interp(sorted_S, bin_centers, bin_means)
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted
        
        # Discount
        cont_values *= discount_factor
        
        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
        # Also store the terminal step table for consistency
        if t == n_dates - 1:
             # For the last step, the "continuation" is 0, but we are at maturity.
             # Actually, the loop runs from n_dates-2 down to 0.
             # We need a table for t = n_dates-1? No, exercise at T is automatic if in the money.
             # But the policy object might be called for time_index = n_dates-1?
             # The task says "12 monthly exercise dates". If n_dates=12, indices 0..11.
             # Usually, exercise at T is not a decision, it's the payoff.
             # However, the policy interface `exercise` takes `time_index`.
             # If time_index is the last one, we should probably exercise if S < K.
             # Let's create a simple table for the last step: threshold is K.
             interp_tables[n_dates-1] = (np.array([0.0, K, K+1]), np.array([1000, 0, 0])) # Dummy, handled separately

    # Determine thresholds by reconstructing the value function on a fine grid
    thresholds = np.zeros(n_dates)
    
    # Fine grid for threshold search
    # Stock prices can go above K, but for a put, threshold is <= K.
    # We search in [0, K].
    grid_S = np.linspace(0, K, 1000)
    
    for t in range(n_dates):
        if _time.time() > deadline - 5.0:
             # Emergency fallback for threshold calculation
             thresholds[t] = K * 0.8
             continue

        if t == n_dates - 1:
            # At maturity, exercise if S < K
            thresholds[t] = K
            continue
            
        if interp_tables[t] is None:
            # Should not happen if loop completed
            thresholds[t] = K * 0.5
            continue
            
        bin_centers, bin_values = interp_tables[t]
        
        # Interpolate the continuation value onto the fine grid
        # Note: bin_values are E[V_{t+1}], so we must discount them to get C_t(S)
        cont_grid = np.interp(grid_S, bin_centers, bin_values) * discount_factor
        
        # Payoff if exercised now
        payoff_grid = np.maximum(K - grid_S, 0.0)
        
        # Difference: Payoff - Continuation
        # Exercise is optimal if Payoff > Continuation
        # i.e., payoff_grid - cont_grid > 0
        diff = payoff_grid - cont_grid
        
        # Find the largest S where diff > 0
        # Since payoff decreases with S and continuation generally increases (or stays flat),
        # the region where diff > 0 is an interval [0, threshold].
        exercise_mask = diff > 1e-5  # Small tolerance to avoid noise at the boundary
        
        if np.any(exercise_mask):
            # Find the last index where exercise is optimal
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            # Interpolate the threshold between last_exercise_idx and last_exercise_idx + 1
            # If it's the last point in the grid, threshold is grid_S[-1]
            if last_exercise_idx < len(grid_S) - 1:
                S1 = grid_S[last_exercise_idx]
                S2 = grid_S[last_exercise_idx + 1]
                D1 = diff[last_exercise_idx]
                D2 = diff[last_exercise_idx + 1]
                # Linear interpolation for the root
                if D1 != D2:
                    thresholds[t] = S1 + (0 - D1) * (S2 - S1) / (D2 - D1)
                else:
                    thresholds[t] = S1
            else:
                thresholds[t] = grid_S[-1]
        else:
            # No exercise region found? This shouldn't happen for a put with S=0.
            # If it does, set threshold to 0.
            thresholds[t] = 0.0
            
        # Ensure threshold is not negative and not greater than K (for a put, exercising above K is suboptimal)
        thresholds[t] = np.clip(thresholds[t], 0.0, K)

    return ExercisePolicy(thresholds, r, dt)

# EVOLVE-BLOCK-END
