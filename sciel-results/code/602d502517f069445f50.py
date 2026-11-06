"""Combined Bermudan put exercise policy: quantile-binned backward induction with refined grid."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a quantile grid.
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
        # Fallback: shouldn't happen, but handle gracefully
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

    # Use quantile-based binning for better coverage
    n_bins = 64

    # Compute bin edges from training data using quantiles
    # Use all training data to determine range
    all_prices = training_paths.flatten()
    # Filter out non-positive values for quantile computation
    positive_prices = all_prices[all_prices > 0]
    if len(positive_prices) < n_bins:
        price_min = max(0.01, all_prices.min() * 0.9 if all_prices.size > 0 else 0.01)
        price_max = all_prices.max() * 1.1 if all_prices.size > 0 else 200.0
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    else:
        # Use quantiles for better spacing
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(positive_prices, quantiles)
        # Ensure strictly increasing
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i-1]:
                bin_edges[i] = bin_edges[i-1] + 1e-6
        # Extend edges slightly
        bin_edges[0] = max(0.01, bin_edges[0] * 0.95)
        bin_edges[-1] = bin_edges[-1] * 1.05

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Backward induction
    # V_next will hold V_{i+1}(S_{i+1}) for each path
    # Start at last exercise date: V = max(K - S, 0)
    V_next = np.maximum(K - training_paths[:, -1], 0.0)

    # Store exercise decision boundaries: for each date, store the threshold
    # Actually, we need a policy that maps (S, date_index) -> exercise or not
    # Use piecewise constant approximation: for each date and bin, store exercise decision
    # But since the optimal policy is monotone (exercise for low S), we can store a threshold

    # Store continuation value estimates as bin averages for each date
    # cont_bins[i][b] = estimated E[V_{i+1} | S_i in bin b]
    # Then exercise if K - S >= disc * cont_bins[i][bin(S)]

    # For efficiency, store the continuation value per bin per date
    # And also store the exercise threshold per date

    # Let's compute backward and store the continuation value function as bin averages
    # For the policy, we'll store for each date a mapping from bin to continuation value
    # Then at evaluation time, bin the input and compare K-S vs disc*cont

    # cont_value_by_date[i] = array of shape (n_bins,) with E[V_{i+1} | S_i in bin]
    # For the last date, there's no continuation (we always exercise if ITM)
    # Actually for date i (0-indexed), continuation is based on V_{i+1}
    # For date n_dates-1 (last), V = max(K-S, 0), no continuation needed

    # Let's store: for each date i from 0 to n_dates-2, the continuation value per bin
    # For date n_dates-1, always exercise if ITM

    # cont_per_date[i] = array of shape (n_bins,) for i in 0..n_dates-2
    cont_per_date = [None] * (n_dates - 1)

    # We also need to compute V_i for backward induction
    # V_values[i] = V_i(S_i) for each path, used to compute V_{i-1}
    V_current = np.maximum(K - training_paths[:, -1], 0.0)  # V at last date

    # Process dates from n_dates-2 down to 0
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5:
            # Break early, use default policy for remaining dates
            break

        S_i = training_paths[:, i]

        # Bin the S_i values
        bin_indices = np.digitize(S_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute mean V_current (which is V_{i+1}) for each bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, V_current)
        np.add.at(bin_counts, bin_indices, 1)

        bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
        bin_means = bin_sums / bin_counts_safe

        # For bins with no data, interpolate or use overall mean
        if np.any(bin_counts == 0):
            overall_mean = np.mean(V_current)
            # Linear interpolation for empty bins
            valid_mask = bin_counts > 0
            if np.sum(valid_mask) >= 2:
                valid_indices = np.where(valid_mask)[0]
                valid_values = bin_means[valid_mask]
                # Interpolate
                bin_means = np.interp(np.arange(n_bins), valid_indices, valid_values)
            else:
                bin_means = np.where(bin_counts > 0, bin_means, overall_mean)

        cont_per_date[i] = bin_means.copy()

        # Get continuation value for each path
        cont_value = bin_means[bin_indices]
        continue_value = disc * cont_value

        # Exercise payoff
        exercise_payoff = np.maximum(K - S_i, 0.0)

        # Optimal value
        V_i = np.maximum(exercise_payoff, continue_value)

        # Update V_current for next iteration
        V_current = V_i

    # Build the policy object
    # ExercisePolicy: exercise(current_state, time) -> boolean vector
    # current_state: array of spot prices (one per path)
    # time: date index (0 to n_dates-1)
    # Return: boolean array, True if exercise

    class ExercisePolicy:
        def __init__(self, bin_edges, cont_per_date, K, disc, n_dates):
            self.bin_edges = bin_edges
            self.cont_per_date = cont_per_date
            self.K = K
            self.disc = disc
            self.n_dates = n_dates
            self.n_bins = len(bin_edges) - 1

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
            n = len(current_state)

            if time >= self.n_dates - 1:
                # Last exercise date: exercise if in-the-money
                return current_state < self.K

            # Bin the states
            bin_indices = np.digitize(current_state, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

            # Get continuation value for each state
            cont_values = self.cont_per_date[time][bin_indices]
            continue_value = self.disc * cont_values

            # Exercise payoff
            exercise_payoff = np.maximum(self.K - current_state, 0.0)

            # Exercise if exercise payoff >= continuation value
            return exercise_payoff >= continue_value

    return ExercisePolicy(bin_edges, cont_per_date, K, disc, n_dates)
# EVOLVE-BLOCK-END
