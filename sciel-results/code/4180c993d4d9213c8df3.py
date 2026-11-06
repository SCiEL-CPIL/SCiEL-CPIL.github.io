"""Bermudan exercise-policy: Longstaff-Schwartz regression-based backward induction with Quantile Regression."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_train = training_paths.shape[0]
    n_steps = n_dates  # monthly steps

    # Exercise times (no time-0 exercise, so dates are at T/12, 2T/12, ..., 12T/12)
    dt = T / n_steps
    exercise_times = np.linspace(dt, T, n_steps)

    # Discount factors
    discount = np.exp(-r * exercise_times)

    # Generate training paths if not provided, or use provided
    # training_paths shape: (n_train, n_steps) - stock prices at each exercise date
    if training_paths is None or len(training_paths) == 0:
        # Generate antithetic pairs
        rng = np.random.default_rng(42)
        half = n_train // 2
        z = rng.standard_normal((half, n_steps))
        z_anti = -z
        z_all = np.vstack([z, z_anti])
        log_drift = (r - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)
        log_S = np.log(S0) + np.cumsum(log_drift + vol_step * z_all, axis=1)
        training_paths = np.exp(log_S)
        n_train = training_paths.shape[0]

    # Ensure we have enough paths
    n_train = training_paths.shape[0]

    # Initialize continuation values at final step
    # V_final = max(K - S_final, 0)
    S_final = training_paths[:, -1]
    V_next = np.maximum(K - S_final, 0.0)

    # Store fitted regression parameters for each step
    # We will use piecewise linear regression based on quantiles
    # reg_params[i] = (x_edges, y_means)
    reg_params = [None] * n_steps
    n_bins = 20  # Number of quantile bins

    # Backward induction
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Not enough time, use simple threshold for remaining steps
            break

        S_i = training_paths[:, i]
        
        # Compute quantiles for S_i
        # Use np.quantile to get bin edges
        # We want bins that cover the range of S_i
        q = np.quantile(S_i, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges to avoid issues with np.searchsorted
        # Add small epsilon to edges to handle duplicates
        q[1:-1] += 1e-9
        
        # Assign each S_i to a bin
        bin_indices = np.searchsorted(q, S_i, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate mean V_next for each bin
        y_means = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                y_means[b] = np.mean(V_next[mask])
                counts[b] = np.sum(mask)
            else:
                # If a bin is empty, interpolate or use 0
                # For puts, high S -> low value. Low S -> high value.
                # We can just leave it as 0 or interpolate later.
                # Better: use the mean of the whole V_next if empty? No, that's bad.
                # We'll handle empty bins by linear interpolation in the policy or just 0.
                # Given 4096 paths and 20 bins, empty bins are unlikely for central S.
                pass

        # Store edges and means
        # We need to evaluate the piecewise linear function.
        # x_edges are the bin centers or edges?
        # np.interp expects xp (increasing) and fp.
        # Let's use the bin centers as x coordinates and y_means as y.
        bin_centers = (q[:-1] + q[1:]) / 2.0
        
        # Handle potential empty bins by filling y_means with interpolated values
        # If a bin is empty, we can't compute its mean.
        # We can fill empty bins by linear interpolation of the non-empty ones.
        valid_mask = counts > 0
        if not np.all(valid_mask):
            # Interpolate y_means for empty bins
            # Use np.interp on the valid points
            valid_x = bin_centers[valid_mask]
            valid_y = y_means[valid_mask]
            y_means = np.interp(bin_centers, valid_x, valid_y)

        reg_params[i] = (bin_centers, y_means)

    # Now create the exercise policy
    # At each step i, exercise if (K - S_i) >= exp(-r*dt) * fitted_continuation(S_i)
    # i.e., exercise if (K - S_i) >= discount[i] * interp(S_i)

    class BermudanExercisePolicy:
        def __init__(self, reg_params, discount, dt, K, n_steps):
            self.reg_params = reg_params
            self.discount = discount
            self.dt = dt
            self.K = K
            self.n_steps = n_steps
            self.current_step = 0

        def exercise(self, current_state, time):
            """
            current_state: array of stock prices
            time: current time (to determine which exercise date)
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2dt, ..., n_steps*dt
            # time should match one of these
            if time is None or time <= 0:
                return np.zeros(len(current_state), dtype=bool)

            # Find the step index
            step_idx = int(round(time / self.dt)) - 1
            step_idx = max(0, min(step_idx, self.n_steps - 1))

            # Get the regression parameters for this step
            if step_idx < len(self.reg_params) and self.reg_params[step_idx] is not None:
                x_centers, y_means = self.reg_params[step_idx]
            else:
                # Fallback: use simple threshold
                # For a put, exercise when S is low enough
                # Approximate threshold: exercise when S < K * 0.85
                return current_state < self.K * 0.85

            # Evaluate continuation value using interpolation
            # np.interp returns the linear interpolation of y for given x
            cont_value = np.interp(current_state, x_centers, y_means)

            # Discounted continuation value
            disc_cont = self.discount[step_idx] * cont_value

            # Immediate exercise payoff
            exercise_payoff = np.maximum(self.K - current_state, 0.0)

            # Exercise if immediate payoff >= discounted continuation
            return exercise_payoff >= disc_cont

    # Build the policy
    policy = BermudanExercisePolicy(reg_params, discount, dt, K, n_steps)
    return policy

# EVOLVE-BLOCK-END
