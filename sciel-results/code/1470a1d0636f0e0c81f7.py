"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.ndimage import gaussian_filter1d

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
    
    # Backward Induction with Kernel Smoothing (Bin + Gaussian Filter)
    
    # Parameters for smoothing
    n_bins = 200
    kernel_sigma = 5.0  # Smoothness in bin units
    
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 15.0:
            # Fallback: simple heuristic
            # For puts, threshold is typically lower early on
            for t_rem in range(t, -1, -1):
                time_frac = t_rem / n_dates
                # Approximate threshold
                thresholds_rem = K * (0.8 - 0.2 * time_frac)
                # We need to set thresholds for the return object later
                # But we haven't created the object yet. We'll just break and handle below.
                pass
            # If we break here, we need to ensure thresholds are set.
            # Let's just finish the loop quickly with a rough estimate
            for t_rem in range(t, -1, -1):
                S_t = training_paths[t_rem]
                payoff_now = np.maximum(K - S_t, 0.0)
                # Rough continuation: discounted terminal value approximation
                # This is just a fallback to avoid crash
                cont_val = discount_factor * np.mean(V[-1]) 
                V[t_rem] = np.maximum(payoff_now, cont_val)
            break
        
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # 1. Sort S_t and V_next
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # 2. Bin into quantiles
        # Use np.quantile to get edges
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(sorted_S, quantiles)
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            cont_val = discount_factor * np.mean(V_next)
            cont_values = np.full(n_paths, cont_val)
            payoff_now = np.maximum(K - S_t, 0.0)
            V[t] = np.maximum(payoff_now, cont_values)
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
            # All empty? Degenerate
            bin_means[:] = np.mean(V_next)
            
        # 5. Smooth the bin means using Gaussian filter
        # This enforces smoothness in the continuation value function
        smoothed_bin_means = gaussian_filter1d(bin_means, sigma=kernel_sigma, mode='nearest')
        
        # 6. Interpolate back to original S_t
        # We need the center of each bin for interpolation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Interpolate smoothed values at the sorted S_t locations
        cont_values_sorted = np.interp(sorted_S, bin_centers, smoothed_bin_means)
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted
        
        # Discount
        cont_values *= discount_factor
        
        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
    # Determine thresholds
    thresholds = np.zeros(n_dates)
    
    for t in range(n_dates):
        S_t = training_paths[t]
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Exercise is optimal when V[t] == payoff_now (and payoff > 0)
        # Due to floating point, V[t] is max(payoff, cont).
        # So exercise if payoff >= cont.
        # We can check where V[t] - payoff_now is close to 0?
        # Better: Recalculate cont? No, we didn't store it.
        # But V[t] = max(payoff, cont).
        # If payoff > cont, V[t] = payoff.
        # If payoff <= cont, V[t] = cont.
        # So exercise region is where V[t] == payoff_now (within tolerance).
        
        # To be robust to noise in the threshold, we can look for the boundary.
        # Since the policy is a threshold, we want the max S where exercise is optimal.
        # Let's find all indices where exercise was optimal in training.
        tol = 1e-5
        is_exercise = (np.abs(V[t] - payoff_now) < tol) & (payoff_now > 0)
        
        if np.any(is_exercise):
            # Use the 95th percentile of the exercise prices to be robust against outliers/noise
            # Or just the max. Max is theoretically correct if the estimator is unbiased.
            # But noise might cause a high S to be incorrectly classified as exercise.
            # Using a high percentile (e.g. 99th) of the exercise set is safer.
            ex_prices = S_t[is_exercise]
            # If very few points, just use max.
            if len(ex_prices) > 10:
                thresholds[t] = np.percentile(ex_prices, 99)
            else:
                thresholds[t] = np.max(ex_prices)
        else:
            # No exercise in training
            # Conservative threshold
            time_frac = t / n_dates
            thresholds[t] = K * (0.5 + 0.4 * time_frac)
            
    return ExercisePolicy(thresholds, r, dt)

# EVOLVE-BLOCK-END
