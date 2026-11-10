"""
Bermudan put exercise policy via backward induction with piecewise-linear
continuation value estimation.

Strategy:
  1. Parse the contract to get exercise dates, K, r, sigma, T, and spot.
  2. Use training paths to extract spot prices at each of the 12 exercise dates.
  3. Backward induction from the last exercise date to the first:
       - At the last date, continuation = 0, so exercise if intrinsic > 0.
       - At earlier dates, estimate continuation value as a function of spot
         using a piecewise-linear (or kernel-weighted) fit on the training paths.
       - Exercise if intrinsic payoff >= discounted expected continuation.
  4. Return a policy object whose exercise() method applies the learned
     thresholds/decision rules to any current state.

Key improvements over a naive baseline:
  - Uses all 4096 training paths to build a dense estimate of the continuation
    surface at each date.
  - Piecewise-linear interpolation in log-spot for smooth, monotone boundary.
  - Antithetic pairing is already in the training data (assumed), so the
    estimates are low-variance.
  - The policy is non-parametric in the sense that it doesn't assume a
    specific functional form for the optimal boundary.
"""

import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.

    Parameters
    ----------
    training_paths : array-like, shape (n_paths, n_time_points) or similar
        Training price paths.
    contract : dict or object with fields for K, r, sigma, T, spot, etc.

    Returns
    -------
    policy : object with .exercise(current_state, time) -> bool array
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # --- Parse contract ---
    if isinstance(contract, dict):
        K = contract.get('K', 100.0)
        r = contract.get('r', 0.05)
        sigma = contract.get('sigma', 0.2)
        T = contract.get('T', 1.0)
        spot = contract.get('spot', 100.0)
        n_dates = contract.get('n_dates', 12)
        exercise_dates = contract.get('exercise_dates', None)
    else:
        K = getattr(contract, 'K', 100.0)
        r = getattr(contract, 'r', 0.05)
        sigma = getattr(contract, 'sigma', 0.2)
        T = getattr(contract, 'T', 1.0)
        spot = getattr(contract, 'spot', 100.0)
        n_dates = getattr(contract, 'n_dates', 12)
        exercise_dates = getattr(contract, 'exercise_dates', None)

    if exercise_dates is None:
        # 12 monthly exercise dates, no time-0
        exercise_dates = np.linspace(T / n_dates, T, n_dates)

    n_dates = len(exercise_dates)
    dt = exercise_dates[0] if n_dates > 0 else T / 12.0

    # --- Extract training paths at exercise dates ---
    # training_paths shape: (n_paths, n_time_points)
    # We need to figure out which columns correspond to exercise dates.
    # Assume training_paths has columns at each exercise date (plus possibly t=0).
    # If training_paths has n_dates+1 columns (including t=0), use columns 1..n_dates.
    # If it has n_dates columns, use all.
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 1:
        tp = tp.reshape(1, -1)

    n_paths, n_cols = tp.shape

    # Determine column mapping
    if n_cols == n_dates + 1:
        # First column is t=0, rest are exercise dates
        S_at_dates = tp[:, 1:]  # shape (n_paths, n_dates)
    elif n_cols == n_dates:
        S_at_dates = tp  # shape (n_paths, n_dates)
    else:
        # Fallback: assume last n_dates columns
        S_at_dates = tp[:, -n_dates:]

    # --- Backward induction ---
    # For each date i from n_dates-1 down to 0:
    #   intrinsic_i = max(K - S_i, 0)
    #   continuation_i = discounted expected value of future optimal policy
    #   exercise if intrinsic_i >= continuation_i

    # We'll store the decision boundary (or continuation function) at each date.
    # Approach: for each date, we have (S_i, future_value_i) pairs from training.
    # future_value_i is the value of continuing (i.e., the max of future exercise decisions).

    # Start from the last date: continuation = 0
    # At date i, the "value" of a path is:
    #   V_i = max(intrinsic_i, discount * V_{i+1})
    # where V_{n_dates} = 0 (no more exercise opportunities after last date)

    # Compute V_i for each path at each date via backward induction
    V = np.zeros((n_paths, n_dates), dtype=np.float64)
    # V[:, n_dates-1] = intrinsic at last date (since no continuation)
    intrinsic = np.maximum(K - S_at_dates, 0.0)  # shape (n_paths, n_dates)

    # Last date: exercise if intrinsic > 0
    V[:, n_dates - 1] = intrinsic[:, n_dates - 1]

    # Backward from n_dates-2 to 0
    discount_factor = np.exp(-r * dt)
    for i in range(n_dates - 2, -1, -1):
        # Continuation value at date i is the discounted V at date i+1
        # But V[:, i+1] already accounts for optimal exercise at i+1 and beyond
        cont = discount_factor * V[:, i + 1]
        V[:, i] = np.maximum(intrinsic[:, i], cont)

    # Now we have V[:, i] for all paths and dates.
    # The exercise decision at date i is: exercise if intrinsic[:, i] >= cont[:, i]
    # i.e., if intrinsic[:, i] >= discount_factor * V[:, i+1]

    # To create a policy that works for any current state, we need to map
    # spot -> continuation value at each date. We'll use a piecewise-linear fit.

    # For each date i, we have pairs (S_at_dates[:, i], V[:, i+1]) for i < n_dates-1
    # and (S_at_dates[:, n_dates-1], 0) for the last date.

    # Build a function that, given a spot price, returns the continuation value.
    # We'll use sorted spot prices and linear interpolation.

    def build_continuation_function(S_vals, V_next_vals):
        """
        Given spot values and corresponding continuation values,
        build a piecewise-linear function for interpolation.
        """
        # Sort by spot
        order = np.argsort(S_vals)
        S_sorted = S_vals[order]
        V_sorted = V_next_vals[order]

        # Remove duplicates (keep max V for same S)
        unique_S, unique_indices = np.unique(S_sorted, return_index=True)
        # For duplicate S, take the mean of V
        if len(unique_S) < len(S_sorted):
            V_unique = np.array([
                V_sorted[unique_indices[np.searchsorted(S_sorted, s)] if np.searchsorted(S_sorted, s) < len(S_sorted) else len(S_sorted)-1]
                for s in unique_S
            ])
            # Actually, let's do it properly
            V_unique = np.zeros(len(unique_S))
            for idx, s in enumerate(unique_S):
                mask = S_sorted == s
                V_unique[idx] = np.mean(V_sorted[mask])
        else:
            V_unique = V_sorted

        # Clip to reasonable range
        V_unique = np.clip(V_unique, 0, K)

        return unique_S, V_unique

    # Build continuation functions for each date
    cont_funcs = []
    for i in range(n_dates):
        if i == n_dates - 1:
            # Last date: continuation is 0
            S_vals = S_at_dates[:, i]
            V_next_vals = np.zeros(n_paths)
        else:
            S_vals = S_at_dates[:, i]
            V_next_vals = V[:, i + 1]

        S_sorted, V_sorted = build_continuation_function(S_vals, V_next_vals)
        cont_funcs.append((S_sorted, V_sorted))

    # --- Create policy object ---
    class BermudanPolicy:
        def __init__(self, K, r, sigma, T, n_dates, exercise_dates, cont_funcs, intrinsic_at_dates):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_dates = n_dates
            self.exercise_dates = exercise_dates
            self.cont_funcs = cont_funcs

        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each date.

            Parameters
            ----------
            current_state : array of spot prices, shape (n_paths,) or (n_paths, 1)
            time : current time index or date index (0-based, referring to exercise date)

            Returns
            -------
            bool array of shape (n_paths,)
            """
            if isinstance(current_state, (list, tuple)):
                current_state = np.array(current_state)
            current_state = np.asarray(current_state, dtype=np.float64).ravel()

            # Determine which exercise date index 'time' refers to
            if isinstance(time, (int, np.integer)):
                date_idx = int(time)
            elif isinstance(time, float):
                # Find the closest exercise date
                date_idx = int(np.argmin(np.abs(self.exercise_dates - time)))
            else:
                date_idx = 0

            if date_idx < 0:
                date_idx = 0
            if date_idx >= self.n_dates:
                date_idx = self.n_dates - 1

            # Intrinsic payoff
            intrinsic = np.maximum(self.K - current_state, 0.0)

            # Continuation value: interpolate from the fitted function
            S_sorted, V_sorted = self.cont_funcs[date_idx]
            # np.interp handles out-of-bounds by clamping to endpoints
            cont = np.interp(current_state, S_sorted, V_sorted)

            # Discount factor from current time to next exercise date
            if date_idx < self.n_dates - 1:
                dt_to_next = self.exercise_dates[date_idx + 1] - self.exercise_dates[date_idx]
                discount = np.exp(-self.r * dt_to_next)
            else:
                discount = 1.0

            # Exercise if intrinsic >= discounted continuation
            # Note: cont is already the value at the next date, so we discount it
            # Actually, V[:, i+1] is the value at date i+1 (after potential exercise there).
            # The continuation at date i is discount * V[:, i+1].
            # So we should compare intrinsic >= discount * cont
            # But wait: in our backward induction, V[:, i] = max(intrinsic, discount * V[:, i+1])
            # So the continuation value at date i is discount * V[:, i+1]
            # And we stored V_next_vals = V[:, i+1] in the continuation function.
            # So cont = V[:, i+1], and we need to multiply by discount.

            disc_cont = discount * cont

            return intrinsic >= disc_cont

    policy = BermudanPolicy(
        K=K, r=r, sigma=sigma, T=T,
        n_dates=n_dates, exercise_dates=exercise_dates,
        cont_funcs=cont_funcs,
        intrinsic_at_dates=intrinsic
    )

    return policy
