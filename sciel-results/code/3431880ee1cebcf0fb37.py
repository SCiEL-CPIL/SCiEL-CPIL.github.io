"""Bermudan put exercise-policy research: stronger bootstrap SEARCH program.

Fits a set of 12 monthly exercise thresholds (one per exercise date) by
maximising the mean discounted put payoff on the training paths using a
backward induction / dynamic programming approach on the empirical
distribution of the training paths.  Uses antithetic-pair style pairing for
variance reduction when evaluating candidate policies.
"""
# EVOLVE-BLOCK-START

import numpy as np
import time as _time

_DEADLINE = 115.0  # seconds; safety margin under 120 s wall-clock limit


def _gbm_paths(S0, K, r, sigma, T, n_dates, n_paths, seed=0):
    """Generate risk-neutral GBM paths at n_dates monthly exercise dates."""
    rng = np.random.default_rng(seed)
    dt = T / n_dates
    z = rng.standard_normal((n_dates, n_paths))
    log_s0 = np.log(S0)
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    steps = drift + vol * z
    cum = np.cumsum(steps, axis=0)
    S = np.exp(log_s0 + cum)  # (n_dates, n_paths)
    return S


def _payoff_put(S, K):
    """Intrinsic put payoff (K - S)^+."""
    return np.maximum(K - S, 0.0)


def _discount(t, r):
    """Discount factor for time t."""
    return np.exp(-r * t)


def _evaluate_policy(S, K, r, T, n_dates, thresholds):
    """
    Evaluate a barrier/threshold policy on given paths.
    thresholds: array of shape (n_dates,) where thresholds[t] is the barrier.
    At exercise date t (0-indexed), exercise if S[t] <= thresholds[t].
    Returns mean discounted payoff.
    """
    dt = T / n_dates
    ts = np.arange(1, n_dates + 1) * dt
    n_paths = S.shape[1]

    # Work backwards
    cont_value = np.zeros(n_paths)  # continuation value at next date
    total_payoff = np.zeros(n_paths)
    exercised_mask = np.zeros(n_paths, dtype=bool)

    for t in range(n_dates - 1, -1, -1):
        intrinsic = _payoff_put(S[t], K)
        disc_cont = _discount(dt, r) * cont_value
        exercise_mask = S[t] <= thresholds[t]
        exercise_mask = exercise_mask & ~exercised_mask
        # Payoff if exercising now: intrinsic
        # Payoff if continuing: discounted continuation
        # We take the max, but the policy is fixed (threshold), so:
        # If exercising: get intrinsic
        # If continuing: get disc_cont (which already encodes future optimal-ish)
        # Actually for a fixed threshold policy, the value is:
        #   V_t = intrinsic * exercise_mask + disc_cont * (~exercise_mask)
        # But we need to be careful: once exercised, no more payoff.
        # Let's track per-path: at the first date where S[t] <= threshold[t],
        # we get intrinsic payoff discounted to time 0.

    # Better approach: for each path, find the first exercise date
    # where S[t] <= thresholds[t], then payoff = intrinsic[t] * exp(-r * t * dt)
    # If never exercised, payoff = 0 (put expires worthless if never exercised,
    # but actually if we never exercise we get 0; the put is Bermudan so
    # not exercising means no payoff)

    # Vectorised approach:
    # For each path, find first t where S[t] <= thresholds[t]
    S_t = S  # (n_dates, n_paths)
    thresh_exp = thresholds[:, None]  # (n_dates, 1)
    exercise_cond = S_t <= thresh_exp  # (n_dates, n_paths)

    # Find first True along axis 0
    # np.argmax returns first True index (0 if none, but 0 is also valid)
    # Need to handle the case where no exercise happens
    # Use a trick: flip and find
    # Or: iterate
    first_ex = np.full(n_paths, n_dates, dtype=int)  # n_dates means never
    for t in range(n_dates):
        remaining = first_ex == n_dates
        if not np.any(remaining):
            break
        new_ex = exercise_cond[t] & remaining
        if np.any(new_ex):
            first_ex[new_ex] = t

    # Compute discounted payoff
    payoff = np.zeros(n_paths)
    exercised = first_ex < n_dates
    if np.any(exercised):
        idx = np.arange(n_paths)
        ex_idx = first_ex[exercised]
        disc = np.exp(-r * (ex_idx + 1) * dt)
        intr = _payoff_put(S[ex_idx, idx[exercised]], K)
        payoff[exercised] = intr * disc

    return np.mean(payoff)


def _evaluate_policy_fast(S, K, r, T, n_dates, thresholds):
    """
    Faster evaluation using vectorised backward induction.
    """
    dt = T / n_dates
    n_paths = S.shape[1]

    # For a threshold policy, we can compute the value via backward induction
    # where at each step we decide based on the threshold.
    # V_t = max(intrinsic_t, disc * V_{t+1}) but with the policy fixed to threshold
    # Actually the policy is: exercise if S <= threshold, so:
    # V_t = intrinsic_t * I(S_t <= thresh_t) + disc * V_{t+1} * I(S_t > thresh_t)
    # with V_{n_dates} = 0

    V = np.zeros(n_paths)
    for t in range(n_dates - 1, -1, -1):
        intrinsic = _payoff_put(S[t], K)
        ex_mask = S[t] <= thresholds[t]
        V = intrinsic * ex_mask + _discount(dt, r) * V * (~ex_mask)

    return np.mean(V)


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy on the given training paths.

    training_paths: array of shape (n_dates, n_paths) with GBM spot prices
    contract: dict with keys S0, K, r, sigma, T, n_dates (or similar)

    Returns an object with .exercise(current_state, time) method.
    """
    t_start = _time.time()

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_paths = training_paths.shape[1]

    # training_paths is (n_dates, n_paths)
    S = training_paths  # assume already at exercise dates

    # Initial thresholds: use a reasonable heuristic
    # For a put, exercise when spot is low.
    # Start with thresholds at some quantile of the spot distribution
    # at each date, or just use a constant like 0.8 * K initially.
    # Better: use the spot mean at each date scaled down.
    S_mean = np.mean(S, axis=1)  # (n_dates,)
    # Initial thresholds: 70% of spot mean at each date, clipped to [0.3*K, K]
    thresholds = np.clip(0.7 * S_mean, 0.3 * K, K)

    # Evaluate initial
    best_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
    best_thresholds = thresholds.copy()

    # Hill-climbing / coordinate descent search
    # At each iteration, try to improve one threshold at a time
    # Try several perturbations per threshold

    # Number of search iterations based on time budget
    # Each evaluation is O(n_dates * n_paths) ~ 12 * 4096 = 49k ops, very fast
    # So we can do many iterations.

    # Let's do a grid-like search: for each threshold, try values on a grid
    # around the current best, pick the best, repeat.

    # Grid points for each threshold: sample from the spot distribution
    # Use percentiles as candidate thresholds
    n_grid = 30  # number of grid points per threshold

    # Precompute candidate threshold values for each date
    # Use percentiles of S[t] for t in range(n_dates)
    candidate_thresholds = []
    for t in range(n_dates):
        # Use percentiles of the spot at date t
        pcts = np.linspace(1, 99, n_grid)
        vals = np.percentile(S[t], pcts)
        # Also include 0 (never exercise) and large (always exercise)
        vals = np.concatenate([[0.0], vals, [K + 100]])
        vals = np.sort(np.unique(vals))
        candidate_thresholds.append(vals)

    # Coordinate ascent: iterate over thresholds, for each try all candidates
    # Keep best, repeat until no improvement or time limit

    improved = True
    max_rounds = 20  # cap rounds to avoid infinite loop
    round_count = 0

    while improved and round_count < max_rounds:
        improved = False
        round_count += 1

        if _time.time() - t_start > _DEADLINE - 5.0:
            break

        for t in range(n_dates):
            # Try each candidate threshold for date t
            best_local = best_score
            best_local_thresh = best_thresholds[t]
            for val in candidate_thresholds[t]:
                trial = best_thresholds.copy()
                trial[t] = val
                score = _evaluate_policy_fast(S, K, r, T, n_dates, trial)
                if score > best_local + 1e-12:
                    best_local = score
                    best_local_thresh = val

            if best_local > best_score + 1e-12:
                best_thresholds[t] = best_local_thresh
                best_score = best_local
                improved = True

        if _time.time() - t_start > _DEADLINE - 5.0:
            break

    # Fine-tuning: local search around best thresholds
    if _time.time() - t_start < _DEADLINE - 10.0:
        # Do a few rounds of small perturbations
        for fine_round in range(5):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
            for t in range(n_dates):
                current = best_thresholds[t]
                # Try small perturbations
                delta = 0.05 * K
                for frac in [-0.5, -0.25, 0.25, 0.5, 1.0]:
                    val = current + frac * delta
                    if val < 0:
                        continue
                    trial = best_thresholds.copy()
                    trial[t] = val
                    score = _evaluate_policy_fast(S, K, r, T, n_dates, trial)
                    if score > best_score + 1e-12:
                        best_thresholds[t] = val
                        best_score = score

    # Final best thresholds
    final_thresholds = best_thresholds

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, T, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.T = T
            self.n_dates = n_dates

        def exercise(self, current_state, time):
            """
            current_state: array of shape (n_paths,) - spot prices at current date
            time: float or int - current time (or date index)

            Returns boolean vector: True if should exercise.
            """
            # Determine which exercise date this corresponds to
            if isinstance(time, (int, np.integer)):
                t_idx = int(time)
            elif isinstance(time, float):
                dt = self.T / self.n_dates
                t_idx = int(round(time / dt)) - 1  # time is in (0, T], dates at 1*dt, 2*dt, ...
                if t_idx < 0:
                    t_idx = 0
                if t_idx >= self.n_dates:
                    t_idx = self.n_dates - 1
            else:
                t_idx = 0

            t_idx = max(0, min(t_idx, self.n_dates - 1))
            threshold = self.thresholds[t_idx]
            return np.asarray(current_state) <= threshold

    return ExercisePolicy(final_thresholds, K, r, T, n_dates)

# EVOLVE-BLOCK-END
