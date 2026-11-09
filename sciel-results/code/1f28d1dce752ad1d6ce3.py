"""Bermudan exercise-policy research: fit a policy from training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    # training_paths shape: (N_paths, n_time_points)
    # n_time_points = n_exercise + 1 = 13 (t=0, 1/12, ..., 1)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # We need to determine the optimal exercise policy for a Bermudan put.
    # The value at each exercise date depends on future values (backward induction).
    # Strategy: use the training paths to estimate the continuation value at each
    # exercise date by regressing the discounted future payoff on current state.

    # For a put: payoff = max(K - S, 0)
    # Exercise dates are indices 1..n_exercise (index 0 is t=0, no exercise)
    # At maturity (index n_exercise), the value is the put payoff.

    # Backward induction with Monte Carlo estimation:
    # At each exercise date i (from n_exercise-1 down to 1):
    #   continuation_value[i] = E[discount * max(exercise_value[i+1], continuation_value[i+1]) | S_i]
    #   exercise_value[i] = max(K - S_i, 0)
    #   optimal: exercise if exercise_value[i] >= continuation_value[i]

    # We'll use a simple approach: for each exercise date, estimate the continuation
    # value as a function of S using quantile regression or simple binning.

    # Actually, let's use a more direct approach with the training paths:
    # Simulate the optimal policy using backward induction on the training paths.

    # Step 1: Determine optimal exercise decisions on training paths via backward induction
    # We'll use a simple threshold approach: for each exercise date, find the threshold
    # S* such that it's optimal to exercise if S <= S*.

    # Better approach: use the training paths to do proper backward induction
    # with conditional expectation estimation.

    # Let's use a simple but effective method:
    # 1. For each exercise date from last to first, estimate the continuation value
    #    as a function of current spot using a piecewise linear or polynomial fit.
    # 2. Use this to determine exercise decisions.

    # For simplicity and speed, let's use a binning approach:
    # Divide the spot range into bins and estimate continuation value per bin.

    # First, determine the range of spots in training data
    all_spots = training_paths[:, 1:]  # exclude time 0
    S_min = np.min(all_spots)
    S_max = np.max(all_spots)

    # Use 50 bins for spot
    n_bins = 50
    bin_edges = np.linspace(S_min * 0.5, S_max * 1.5, n_bins + 1)

    # Backward induction
    # V[i, path] = value at exercise date i for path
    # At maturity (i = n_exercise), V = max(K - S, 0)

    V = np.zeros((n_exercise + 1, N))
    # Maturity value
    V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)

    # Thresholds for exercise: exercise if S <= threshold[i]
    thresholds = np.zeros(n_exercise)

    for i in range(n_exercise - 1, -1, -1):
        # Exercise date index in training_paths is i+1 (since index 0 is t=0)
        S_i = training_paths[:, i + 1]
        exercise_payoff = np.maximum(K - S_i, 0)

        # Continuation value: E[discount * V[i+1] | S_i]
        # Estimate using binning
        bin_idx = np.digitize(S_i, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # For each bin, compute mean of discount * V[i+1]
        cont_values = np.zeros(N)
        for b in range(n_bins):
            mask = (bin_idx == b)
            if np.sum(mask) > 0:
                cont_values[mask] = discount * np.mean(V[i + 1][mask])
            else:
                # Fallback: use overall mean
                cont_values[mask] = discount * np.mean(V[i + 1])

        # Total value: max(exercise, continuation)
        V[i] = np.maximum(exercise_payoff, cont_values)

        # Determine threshold: exercise if exercise_payoff >= cont_values
        # Find the largest S where it's optimal to exercise
        # Sort by S and find the boundary
        order = np.argsort(S_i)
        S_sorted = S_i[order]
        ex_sorted = exercise_payoff[order]
        cont_sorted = cont_values[order]

        # Find the largest S where ex >= cont
        should_exercise = ex_sorted >= cont_sorted
        if np.any(should_exercise):
            # Find the last index where it's true
            last_true = np.where(should_exercise)[0][-1]
            thresholds[i] = S_sorted[last_true]
        else:
            thresholds[i] = 0.0  # Never exercise

        # Check time
        if time.time() > deadline - 5:
            break

    # Now we have thresholds for each exercise date.
    # The policy: at exercise date i (1-indexed in the exercise dates),
    # exercise if current_state <= thresholds[i-1]

    # But we need to be careful: the thresholds were estimated on training paths.
    # For evaluation, we just compare current state to threshold.

    # Let's also try to refine using a simple local search on thresholds
    # if we have time

    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a simple refinement: adjust thresholds slightly
        # For each threshold, try small perturbations and see if value improves
        # This is complex, so let's just use the current thresholds

        pass

    # Create the policy object
    class Policy:
        def __init__(self, thresholds, n_exercise):
            self.thresholds = thresholds
            self.n_exercise = n_exercise

        def exercise(self, current_state, time):
            # current_state: array of spot prices
            # time: current time (or index)
            # Return boolean vector: True if should exercise
            # time is likely the exercise date index (0-indexed or 1-indexed)
            # Based on the task: "12 monthly exercise dates (no time-0 exercise)"
            # So time might be 1..12 or 0..11

            # Determine which exercise date we're at
            # If time is continuous, map to exercise date index
            # If time is an integer index, use directly

            if isinstance(time, (int, np.integer)):
                # time is an index
                idx = int(time)
                if idx < 0 or idx >= self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                thresh = self.thresholds[idx]
            else:
                # time is continuous
                # Map to exercise date index
                # Exercise dates are at t = 1/12, 2/12, ..., 12/12
                idx = int(np.floor(time * 12))
                if idx < 1:
                    return np.zeros(len(current_state), dtype=bool)
                idx = idx - 1  # 0-indexed
                if idx < 0 or idx >= self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                thresh = self.thresholds[idx]

            return current_state <= thresh

    return Policy(thresholds, n_exercise)

# EVOLVE-BLOCK-END
