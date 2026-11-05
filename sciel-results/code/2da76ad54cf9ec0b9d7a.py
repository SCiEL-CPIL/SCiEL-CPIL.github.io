"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.signal import savgol_filter

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

def _smooth_continuation(bin_means, n_actual_bins):
    """Apply Savitzky-Golay smoothing to bin means to reduce MC noise."""
    if n_actual_bins < 11:
        return bin_means
    # Use a window that's odd and at least 5, up to min(n_actual_bins, 21)
    win = min(n_actual_bins, 21)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        win = 5
    poly_order = min(4, win - 1)
    try:
        smoothed = savgol_filter(bin_means, window_length=win, polyorder=poly_order)
    except Exception:
        smoothed = bin_means
    return smoothed

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
    
    # Backward Induction with improved binning and smoothing
    n_bins = 80  # Bins for better resolution
    
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 15.0:
            # If running out of time, use simple heuristic for remaining steps
            break
        
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # Estimate E[V_next | S_t] using binning with quantiles
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # Create bin edges using quantiles for roughly equal population per bin
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(sorted_S, quantiles)
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            # Degenerate case: all S_t are the same
            cont_val = discount_factor * np.mean(V_next)
            cont_values = np.full(n_paths, cont_val)
            payoff_now = np.maximum(K - S_t, 0.0)
            V[t] = np.maximum(payoff_now, cont_values)
            continue
        
        # Assign each point to a bin
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # Calculate mean V_next for each bin
        n_actual_bins = len(bin_edges) - 1
        bin_sums = np.zeros(n_actual_bins)
        bin_counts = np.zeros(n_actual_bins)
        
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        np.add.at(bin_counts, bin_indices, 1)
        
        # Avoid division by zero
        valid_bins = bin_counts > 0
        bin_means = np.zeros(n_actual_bins)
        bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
        
        # For bins with no data, use interpolation from neighboring bins
        valid_bin_indices = np.where(valid_bins)[0]
        if len(valid_bin_indices) > 0:
            invalid_bins = np.where(~valid_bins)[0]
            if len(invalid_bins) > 0:
                bin_means[invalid_bins] = np.interp(invalid_bins, valid_bin_indices, bin_means[valid_bin_indices])
        
        # Apply Savitzky-Golay smoothing to reduce MC noise
        bin_means_smoothed = _smooth_continuation(bin_means, n_actual_bins)
        
        # Get continuation values for each path using smoothed means
        cont_values_sorted = bin_means_smoothed[bin_indices]
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted
        
        # Discount the continuation value
        cont_values *= discount_factor
        
        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
    
    # Determine thresholds with a final polish pass
    thresholds = np.zeros(n_dates)
    
    for t in range(n_dates):
        S_t = training_paths[t]
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Find the threshold: maximum S where exercise is optimal
        # Use a slightly relaxed tolerance to capture the boundary more robustly
        tol = 1e-4
        is_exercise = (np.abs(V[t] - payoff_now) < tol) & (payoff_now > 0)
        
        if np.any(is_exercise):
            thresholds[t] = np.max(S_t[is_exercise])
        else:
            # If no exercise in training, use a conservative threshold
            time_frac = t / n_dates
            thresholds[t] = K * (0.5 + 0.4 * time_frac)
    
    return ExercisePolicy(thresholds, r, dt)

# EVOLVE-BLOCK-END
