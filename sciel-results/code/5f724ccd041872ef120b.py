"""Bermudan put exercise policy with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put using backward induction.
    
    Args:
        training_paths: array of shape (n_paths, n_time_steps) with GBM paths
        contract: dict with keys like 'K', 'r', 'sigma', 'T', 'n_exercise_dates', etc.
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = _time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine exercise dates from contract or default to 12 monthly
    n_ex = contract.get('n_exercise_dates', 12)
    # Exercise dates are at 1/12, 2/12, ..., 12/12 (no time-0 exercise)
    dt = T / n_ex
    exercise_times = np.array([(i + 1) * dt for i in range(n_ex)])
    
    # training_paths shape: (n_paths, n_steps) where n_steps includes time 0
    # We need to figure out the time grid
    n_paths, n_steps = training_paths.shape
    
    # The paths should have n_ex + 1 points if they include time 0
    # Or they might be sampled at the exercise dates
    # Let's assume the paths are at times 0, dt, 2*dt, ..., T
    # So n_steps should be n_ex + 1
    
    if n_steps != n_ex + 1:
        # Try to interpolate or resample
        # For now, use as many exercise dates as we can
        n_ex = min(n_ex, n_steps - 1)
        dt = T / n_ex
        exercise_times = np.array([(i + 1) * dt for i in range(n_ex)])
    
    # Backward induction
    # V[i] = value at exercise date i (0-indexed, where i=0 is first exercise date)
    # At maturity (i = n_ex-1), value = max(K - S, 0)
    # At earlier dates, value = max(K - S, exp(-r*dt) * E[V[i+1]])
    
    # We'll estimate the continuation value using the training paths
    # For each exercise date i, we need E[V[i+1] | S_i]
    # This is a regression problem
    
    # Start from maturity
    S_at_maturity = training_paths[:, -1]
    V = np.maximum(K - S_at_maturity, 0.0)
    
    # Store thresholds for exercise policy
    # threshold[i] = critical spot price at exercise date i
    # Exercise if S <= threshold[i]
    thresholds = np.zeros(n_ex)
    
    # At the last exercise date (maturity), always exercise if in the money
    thresholds[n_ex - 1] = K
    
    # Work backwards
    for i in range(n_ex - 2, -1, -1):
        # Check time
        if _time.time() > deadline - 5.0:
            # Not enough time, use simple thresholds
            for j in range(i, -1, -1):
                thresholds[j] = K
            break
        
        S_i = training_paths[:, i]
        
        # We need to estimate E[V | S_i]
        # Use a simple approach: bin the S_i values and compute mean V in each bin
        # Or use a polynomial regression
        
        # For robustness, use a piecewise approach
        # Sort by S_i and compute running averages
        
        # Simple approach: use quantile-based bins
        n_bins = min(50, n_paths // 10)
        if n_bins < 5:
            n_bins = 5
        
        # Create bins based on S_i
        S_sorted = np.sort(S_i)
        bin_edges = np.percentile(S_i, np.linspace(0, 100, n_bins + 1))
        
        # Assign each path to a bin
        bin_indices = np.digitize(S_i, bin_edges[1:-1])
        
        # Compute mean V in each bin
        bin_means = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_means[b] = np.mean(V[mask])
                bin_counts[b] = np.sum(mask)
            else:
                bin_means[b] = np.mean(V)
        
        # Continuation value at S_i: interpolate bin means
        # For each path, get its bin mean as continuation value
        cont_values = bin_means[bin_indices]
        
        # Discounted continuation value
        disc_cont = np.exp(-r * dt) * cont_values
        
        # Exercise value
        exercise_val = np.maximum(K - S_i, 0.0)
        
        # Optimal value: max of exercise and continuation
        V = np.maximum(exercise_val, disc_cont)
        
        # Determine threshold: the spot price where exercise_val = disc_cont
        # For a put, exercise_val = K - S (when S < K), disc_cont is decreasing in S
        # Find the crossing point
        # Simple approach: find the highest S where exercise is optimal
        exercise_mask = exercise_val >= disc_cont
        if np.any(exercise_mask):
            thresholds[i] = np.max(S_i[exercise_mask])
        else:
            thresholds[i] = 0.0
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, thresholds, n_ex):
            self.thresholds = thresholds
            self.n_ex = n_ex
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Args:
                current_state: array of spot prices
                time: current time (scalar)
            
            Returns:
                Boolean array indicating whether to exercise at each path
            """
            # Find which exercise date we're at
            # time should be close to one of the exercise_times
            idx = np.argmin(np.abs(exercise_times - time))
            
            if idx >= self.n_ex:
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[idx]
            return current_state <= threshold
    
    return ExercisePolicy(thresholds, n_ex)
# EVOLVE-BLOCK-END
