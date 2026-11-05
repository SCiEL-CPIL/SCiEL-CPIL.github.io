"""Bermudan put exercise policy via backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths.
    
    Parameters:
    -----------
    training_paths : ndarray of shape (n_dates, n_paths)
        Simulated GBM paths at exercise dates (excluding time 0).
    contract : dict with keys:
        'S0' : initial spot (float)
        'K' : strike price (float)
        'r' : risk-free rate (float)
        'sigma' : volatility (float)
        'T' : maturity (float)
        'n_dates' : number of exercise dates (int, 12)
    
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = _time.time()
    deadline = start_time + 110.0  # safety margin under 120s limit

    # Extract contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # training_paths shape: (n_dates, n_paths)
    # Ensure it's a numpy array
    if not isinstance(training_paths, np.ndarray):
        training_paths = np.array(training_paths)

    n_paths = training_paths.shape[1] if training_paths.ndim == 2 else 4096
    # If paths don't have enough dates, pad or truncate
    if training_paths.shape[0] < n_dates:
        # Pad with NaNs and handle later
        pass
    else:
        training_paths = training_paths[:n_dates]

    # Backward induction
    # V[i] = value at date i (i=0..n_dates-1), where date n_dates-1 is the last
    # Payoff at each date: max(K - S, 0)
    # Discount factor per step
    dt = T / n_dates
    disc = np.exp(-r * dt)

    # Initialize continuation value at the last date to 0 (no further dates)
    # V_last = 0 (after exercise at last date, no more value)
    # Actually, at the last date, you can only exercise, so value = payoff
    # Let's define V[i] as the optimal value at date i BEFORE the decision.
    # At date i, you choose max(payoff_i, disc * E[V[i+1] | S_i])
    # For the last date (i = n_dates-1), V = payoff (no continuation)

    # We'll store thresholds: at each date i, exercise if S_i <= threshold_i
    # For a put, lower spot means higher payoff, so exercise when S is low.

    # Continuation value array
    cont = np.zeros(n_paths)  # continuation value after last date = 0

    # Store thresholds for each date
    thresholds = np.zeros(n_dates)

    # Work backwards from the last date to the first
    for i in range(n_dates - 1, -1, -1):
        # Check time
        if _time.time() > deadline - 5.0:
            # Not enough time, use simple heuristic for remaining dates
            # For puts, exercise if S < K (intrinsic value)
            for j in range(i, -1, -1):
                thresholds[j] = K
            break

        S_i = training_paths[i]
        payoff = np.maximum(K - S_i, 0.0)

        if i == n_dates - 1:
            # Last date: must exercise
            V_i = payoff
        else:
            # Continue: discount expected continuation value
            # We have cont from the next step (already computed)
            # V_i = max(payoff, disc * cont)
            # But cont is the value at date i+1, which we already computed
            # Actually, cont should be the value at date i+1 given S_{i+1}
            # We need to be careful: cont is V[i+1] evaluated at S_{i+1}
            # For backward induction on fixed paths, we just use the path values.
            cont_val = disc * cont
            V_i = np.maximum(payoff, cont_val)

        # Determine threshold: exercise if payoff >= cont_val
        # For puts: exercise if K - S >= cont_val => S <= K - cont_val
        # But cont_val varies by path. We need a single threshold per date.
        # Use the median or a quantile of the boundary.
        # Exercise decision: payoff >= cont_val
        # K - S >= cont_val  =>  S <= K - cont_val
        # The boundary is S_boundary = K - cont_val
        # We want a threshold such that exercise when S <= threshold.
        # Use the median of the implied boundaries where exercise is optimal.
        exercise_mask = (payoff >= cont_val) if i < n_dates - 1 else np.ones(n_paths, dtype=bool)

        if np.any(exercise_mask):
            # Implied boundary: S <= K - cont_val
            # For exercised paths, the boundary is K - cont_val
            boundaries = K - cont_val
            # Use median of boundaries for exercised paths
            # Actually, we want the threshold such that paths with S <= threshold exercise.
            # The natural threshold is the median of S_i for exercised paths,
            # or better: find the boundary where payoff = cont_val.
            # Since cont_val varies, use the median of (K - cont_val) for all paths
            # as a robust threshold.
            thresholds[i] = np.median(boundaries)
        else:
            # No exercise on this path set; set threshold very low
            thresholds[i] = 0.0

        # Update continuation value for next iteration
        cont = V_i

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = len(thresholds)

        def exercise(self, current_state, time):
            """
            Determine which paths to exercise.
            
            Parameters:
            -----------
            current_state : ndarray of shape (n_eval_paths,)
                Current spot prices.
            time : float or int
                Current time or date index.
            
            Returns:
            --------
            boolean array of shape (n_eval_paths,)
            """
            # Determine which date index we're at
            # time could be a float (fraction of T) or an int index
            if isinstance(time, (int, np.integer)):
                idx = int(time)
            else:
                # Assume time is a fraction of T or an absolute time
                # If time < 1, treat as fraction; otherwise as absolute time
                if time <= 1.0:
                    # Fraction of T
                    idx = int(round(time * self.n_dates))
                else:
                    # Absolute time
                    idx = int(round(time / T * self.n_dates))

            # Clamp index
            idx = max(0, min(idx, self.n_dates - 1))

            # Exercise if S <= threshold
            # For a put, lower S means higher payoff
            return current_state <= self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
