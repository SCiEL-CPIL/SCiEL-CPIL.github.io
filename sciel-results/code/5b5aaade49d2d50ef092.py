"""Bermudan exercise-policy research: Quantile-binned Monte Carlo Backward Induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Parse contract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # training_paths is likely an array of shape (n_paths, n_dates) or (n_dates, n_paths)
    # The task says "4096 training paths". Standard convention: rows are paths or columns?
    # Usually, paths are rows. Let's assume shape (n_paths, n_dates).
    # If shape is (n_dates, n_paths), we need to handle that.
    # Let's inspect the input shape to be safe.
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        # Fallback if shape is weird, but contract says 4096 paths.
        # Assume (n_paths, n_dates)
        pass
        
    n_paths, n_steps = paths.shape
    # Ensure n_steps matches n_dates. If not, maybe it's transposed.
    if n_steps != n_dates and n_paths == n_dates:
        paths = paths.T
        n_paths, n_steps = paths.shape

    # Exercise times are 1..12. Index 0 in paths corresponds to t=1?
    # Task: "12 monthly exercise dates (no time-0 exercise)".
    # So paths[:, 0] is S at t=1/12, paths[:, 11] is S at t=12/12.
    
    # Initialize values at the last step (maturity)
    # V_{12} = max(K - S_{12}, 0)
    V_next = np.maximum(K - paths[:, -1], 0.0)
    
    # Storage for continuation value interpolation tables for each time step
    # We need to store the policy for each time step t=1..12.
    # The policy at time t depends on the continuation value from t+1.
    # We will store (bin_edges, bin_means) for the continuation value E[V_{t+1}|S_t]
    # for t=1..11. For t=12, we just use intrinsic value.
    
    # We need to return a policy object.
    # The policy object needs to answer: exercise(S, t) -> bool vector.
    # t is likely the index of the exercise date (1-based or 0-based?).
    # "exercise(current_state, time)".
    # Let's assume time is an integer index 1..12 or 0..11.
    # Given "no time-0 exercise", the first decision is at t=1.
    # I will assume `time` input to `exercise` is 1-indexed (1 to 12) or matches the column index.
    # Let's look at the interface: `exercise(current_state, time)`.
    # If `time` is the step index in the paths array (0 to 11), that's convenient.
    # If `time` is the actual time value (1/12, etc.), we need to map it.
    # Most likely `time` is the integer step index 1..12 or 0..11.
    # Let's assume 1-based indexing for the "date" number, or 0-based for array index.
    # I will design the class to handle both by checking the value range or just assuming 1-based date index.
    # Actually, looking at the baseline `stopping_baseline`, it likely handles the contract specifics.
    # I will assume `time` is the 1-based exercise date (1 to 12).
    
    # Let's refine the binning strategy.
    # Number of bins to start with.
    n_bins = 100
    
    # Data structures for the policy
    # policy_tables[t] = (edges, means) where t is 1-based date index.
    # For t=12, we don't need a table, just intrinsic.
    policy_tables = {}
    
    # Backward induction
    # t goes from n_dates-1 down to 0 (array indices)
    # This corresponds to exercise dates n_dates down to 1.
    # V_next is currently V_{t+1} (values at the next exercise date).
    # We are computing V_t and the policy for date t+1?
    # No.
    # Let's align indices carefully.
    # Array index i corresponds to exercise date i+1.
    # paths[:, i] is S at date i+1.
    # V_next holds the values at date i+2 (or maturity).
    
    # Start with V_next = Intrinsic at date 12.
    # Loop i from 10 down to 0 (array indices for dates 11 down to 1).
    # At step i (date i+1), we have S_i = paths[:, i].
    # We have V_next which are the values at date i+2.
    # We estimate E[V_{i+2} | S_i].
    # Then V_i = max(K - S_i, E[V_{i+2} | S_i]).
    # The exercise decision at date i+1 is: K - S_i > E[V_{i+2} | S_i].
    
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break # Safety break
            
        S_current = paths[:, i]
        
        # Create quantile bins for S_current
        # Ensure S_current has enough unique values
        if np.unique(S_current).size < 10:
            # Fallback to uniform or simple mean if degenerate
            edges = np.linspace(S_current.min(), S_current.max(), n_bins + 1)
        else:
            # Quantile bins
            edges = np.quantile(S_current, np.linspace(0, 1, n_bins + 1))
            # Ensure edges are strictly increasing
            # np.unique helps here
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([S_current.min(), S_current.max()])
            elif len(edges) < n_bins + 1:
                # If too few unique quantiles, pad with linear
                pass # np.unique handles it, but we might have fewer bins. That's fine.
            
        # Assign paths to bins
        # np.digitize returns index 1..len(edges)-1 for values in (edges[0], edges[-1])
        # We want 0..len(edges)-2
        bin_indices = np.digitize(S_current, edges) - 1
        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
        
        # Calculate mean V_next for each bin
        # Use bincount for speed
        total_V = np.bincount(bin_indices, weights=V_next, minlength=len(edges)-1)
        counts = np.bincount(bin_indices, minlength=len(edges)-1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_means = total_V / counts
            bin_means[counts == 0] = 0.0 # Or interpolate later?
            
        # For bins with no data, we might want to interpolate from neighbors.
        # Simple approach: fill with 0 or carry forward.
        # Given GBM, edges are quantiles, so all bins should have data roughly.
        # But if S_current has duplicates, some bins might be empty.
        
        # Store the table for this date (i+1)
        # We need to map S -> Continuation Value
        # We will use np.interp in the exercise function.
        # We need bin centers for interp.
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        
        policy_tables[i+1] = (bin_centers, bin_means)
        
        # Compute V_current for this step
        # For each path, get the continuation value estimate
        # Using interp is more accurate than just bin mean for the *value* calculation,
        # but for the *policy* table, we store the bin means.
        # For the backward step, we need V_current for the next iteration.
        # Let's use the bin mean for simplicity and speed, or interp.
        # Interp is O(N log B). Bin mean is O(1) lookup.
        # Let's use the bin mean for V_current to keep it fast and consistent with the policy table.
        V_cont_est = bin_means[bin_indices]
        
        V_current = np.maximum(K - S_current, V_cont_est)
        
        # Update V_next for the next iteration (which will be for date i)
        V_next = V_current
        
    # Now we have policy_tables for dates 1 to 11.
    # Date 12 is intrinsic.
    
    class BermudanExercisePolicy:
        def __init__(self, tables, K_val, n_dates_val):
            self.tables = tables
            self.K = K_val
            self.n_dates = n_dates_val
            
        def exercise(self, current_state, time):
            # current_state: array of spots
            # time: exercise date index (1-based, 1 to 12)
            # Returns boolean vector
            
            # Handle time indexing
            # If time is 1-based:
            t = int(time)
            
            if t == self.n_dates:
                # Maturity: Exercise if ITM
                return (self.K - current_state) > 0.0
            
            if t in self.tables:
                centers, means = self.tables[t]
                # Interpolate continuation value
                # np.interp requires x to be increasing
                # centers is increasing.
                cont_val = np.interp(current_state, centers, means)
                # Exercise if Intrinsic > Continuation
                return (self.K - current_state) > cont_val
            else:
                # Fallback: Intrinsic (should not happen for t < 12 if tables are built)
                return (self.K - current_state) > 0.0

    return BermudanExercisePolicy(policy_tables, K, n_dates)

# EVOLVE-BLOCK-END
