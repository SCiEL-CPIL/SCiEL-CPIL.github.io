"""Bermudan put exercise policy: Smooth parametric boundary via spline-refined backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math
from scipy.interpolate import UnivariateSpline, make_lsq_spline

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with spline-smoothed 
    continuation values and a parametric smooth exercise boundary.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    # Training paths shape: (n_paths, n_dates)
    if training_paths is None:
        n_paths = 4096
        training_paths = np.zeros((n_paths, n_dates))
    else:
        training_paths = np.asarray(training_paths, dtype=np.float64)
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(-1, n_dates)
        n_paths, actual_n_dates = training_paths.shape
        n_dates = actual_n_dates

    dt = T / n_dates
    disc = math.exp(-r * dt)

    # Use a fine grid for initial binning to reduce bias
    n_bins = 256
    
    # Compute bin edges from training data using quantiles across all dates
    all_prices = training_paths.flatten()
    positive_prices = all_prices[all_prices > 0]
    if len(positive_prices) < n_bins:
        price_min = max(0.01, all_prices.min() * 0.9 if all_prices.size > 0 else 0.01)
        price_max = all_prices.max() * 1.1 if all_prices.size > 0 else 200.0
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(positive_prices, quantiles)
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i-1]:
                bin_edges[i] = bin_edges[i-1] + 1e-6
        bin_edges[0] = max(0.01, bin_edges[0] * 0.95)
        bin_edges[-1] = bin_edges[-1] * 1.05

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Backward induction
    # V_current holds V_{i+1}(S_{i+1}) for each path at the current iteration
    V_current = np.maximum(K - training_paths[:, -1], 0.0)

    # Store estimated boundaries for each date
    boundaries = [None] * (n_dates - 1)

    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5:
            # Break early, use default policy for remaining dates
            # Fill remaining boundaries with last known or simple heuristic
            if boundaries[i+1] is not None:
                 boundaries[i] = boundaries[i+1]
            else:
                 boundaries[i] = K * 0.8 # Fallback
            continue

        S_i = training_paths[:, i]
        S_next = training_paths[:, i+1]

        # Bin the S_next values to compute mean V_current for each bin
        bin_indices_next = np.digitize(S_next, bin_edges) - 1
        bin_indices_next = np.clip(bin_indices_next, 0, n_bins - 1)

        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices_next, V_current)
        np.add.at(bin_counts, bin_indices_next, 1)

        bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
        bin_means = bin_sums / bin_counts_safe

        # Interpolate empty bins
        if np.any(bin_counts == 0):
            valid_mask = bin_counts > 0
            if np.sum(valid_mask) >= 2:
                valid_indices = np.where(valid_mask)[0]
                valid_values = bin_means[valid_mask]
                bin_means = np.interp(np.arange(n_bins), valid_indices, valid_values)
            else:
                overall_mean = np.mean(V_current)
                bin_means = np.where(bin_counts > 0, bin_means, overall_mean)

        # Fit a smooth spline to the bin means to reduce noise
        # We fit V(S) using the bin centers
        valid_bin_mask = bin_counts > 0
        if np.sum(valid_bin_mask) > 3:
            x_spline = bin_centers[valid_bin_mask]
            y_spline = bin_means[valid_bin_mask]
            
            # Sort by x for spline
            sort_idx = np.argsort(x_spline)
            x_spline = x_spline[sort_idx]
            y_spline = y_spline[sort_idx]
            
            # Use a smoothing spline. s parameter controls smoothing.
            # Smaller s = closer to data. Larger s = smoother.
            # n_points ~ 256. s = 0.5 * n_points is a reasonable start.
            s = 0.5 * len(x_spline)
            try:
                spline = UnivariateSpline(x_spline, y_spline, s=s)
                # Evaluate continuation value for current states S_i
                cont_value = spline(S_i)
            except Exception:
                # Fallback to linear interpolation if spline fails
                cont_value = np.interp(S_i, x_spline, y_spline)
        else:
            # Fallback to simple mean if too few bins
            cont_value = np.full_like(S_i, np.mean(V_current))

        continue_value = disc * cont_value
        exercise_payoff = np.maximum(K - S_i, 0.0)

        # Determine optimal decision
        exercise_mask = exercise_payoff >= continue_value
        
        # Estimate boundary B_i
        # The boundary is roughly the max S where we exercise.
        # To be robust, we can take a high percentile of exercised states.
        if np.any(exercise_mask):
            exercised_states = S_i[exercise_mask]
            # Use 95th percentile to avoid outliers
            b_i = np.percentile(exercised_states, 95)
            # Cap boundary at K (exercise only if ITM or close)
            b_i = min(b_i, K)
            boundaries[i] = b_i
        else:
            # If no exercise, boundary is very low
            boundaries[i] = 0.0

        # Update V_current
        V_current = np.where(exercise_mask, exercise_payoff, continue_value)

    # Fit a smooth curve to the boundaries across time
    # We have boundaries[i] for i in 0..n_dates-2
    # Time points are t_i = (i+1) * dt? No, exercise dates are 1..12?
    # Contract says "12 monthly exercise dates (no time-0 exercise)".
    # Usually index 0 corresponds to t=dt, index 11 to t=T.
    # But our loop i goes from n_dates-2 down to 0.
    # boundaries[0] is for date 0 (t=dt). boundaries[n_dates-2] is for date n_dates-2 (t=(n_dates-1)dt).
    # Wait, the last date is n_dates-1. We don't have a boundary for the last date (always exercise if ITM).
    # So we fit B(t) for t in [dt, (n_dates-1)dt].
    
    valid_boundary_indices = [i for i in range(n_dates - 1) if boundaries[i] is not None]
    
    if len(valid_boundary_indices) >= 2:
        t_points = np.array([(i + 1) * dt for i in valid_boundary_indices])
        b_points = np.array([boundaries[i] for i in valid_boundary_indices])
        
        # Sort by time
        sort_idx = np.argsort(t_points)
        t_points = t_points[sort_idx]
        b_points = b_points[sort_idx]
        
        # Fit a smooth spline for the boundary B(t)
        # The boundary for a put typically decreases as t increases (approaching maturity, less time value)
        # Or stays relatively flat. A low-order polynomial or spline is good.
        try:
            # s = 0 for interpolation, larger for smoothing.
            # Use a moderate smoothing to avoid wiggles.
            s = 0.5 * len(t_points)
            boundary_spline = UnivariateSpline(t_points, b_points, s=s)
        except Exception:
            # Fallback to linear interpolation
            boundary_spline = None
    else:
        boundary_spline = None

    class ExercisePolicy:
        def __init__(self, K, n_dates, dt, boundary_spline, boundaries):
            self.K = K
            self.n_dates = n_dates
            self.dt = dt
            self.boundary_spline = boundary_spline
            self.boundaries = boundaries

        def exercise(self, current_state, time):
            """
            Determine exercise decision for given states and time.
            
            Args:
                current_state: array of spot prices
                time: exercise date index (0 to n_dates-1)
            
            Returns:
                Boolean array: True if exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # Last exercise date: exercise if in-the-money
            if time >= self.n_dates - 1:
                return current_state < self.K
            
            # Get the boundary for this time
            t = (time + 1) * self.dt
            
            if self.boundary_spline is not None:
                B_t = self.boundary_spline(t)
            else:
                # Fallback to specific date boundary
                if time < len(self.boundaries) and self.boundaries[time] is not None:
                    B_t = self.boundaries[time]
                else:
                    # Heuristic fallback: exercise if deep ITM
                    B_t = self.K * 0.7

            return current_state < B_t

    return ExercisePolicy(K, n_dates, dt, boundary_spline, boundaries)
# EVOLVE-BLOCK-END
