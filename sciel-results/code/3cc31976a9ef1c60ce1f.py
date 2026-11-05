"""Bermudan put exercise-policy research with robust oracle-threshold fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 115.0  # seconds; safety margin under 120 s wall-clock limit

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt
        self.n_dates = len(thresholds)

    def exercise(self, current_state, time):
        """
        current_state: array of stock prices
        time: integer index of the exercise date (0 to 11) or float time in [0,1]
        Returns: boolean array
        """
        # Determine the index of the exercise date
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
            # If time is a float, assume it's in [0, 1] representing years
            # 12 monthly dates: 1/12, 2/12, ..., 12/12
            # Map to nearest index
            # Index 0 -> 1/12, Index 11 -> 12/12
            # time * 12 gives 1..12. Subtract 1 to get 0..11.
            idx = int(np.round(time * 12)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1

        if idx < 0 or idx >= self.n_dates:
            return np.zeros(len(current_state), dtype=bool)
        
        return current_state < self.thresholds[idx]

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy on the given training paths.
    Uses backward induction to determine optimal exercise decisions for training paths,
    then fits robust thresholds (median of implicit thresholds) for each date.
    """
    t_start = _time.time()

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    S = training_paths  # (n_dates, n_paths)
    if S.ndim != 2:
        # Fallback if shape is unexpected
        S = S.reshape(n_dates, -1)
        
    n_paths = S.shape[1]
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # V[t, i] is the estimated value of the option at time t for path i
    V = np.zeros((n_dates, n_paths))
    thresholds = np.zeros(n_dates)
    
    # Last date: Payoff is intrinsic, exercise if ITM
    V[n_dates-1] = np.maximum(K - S[n_dates-1], 0.0)
    # Implicit threshold at last date is K
    thresholds[n_dates-1] = K
    
    # Backward induction
    for t in range(n_dates - 2, -1, -1):
        if _time.time() - t_start > _DEADLINE:
            # Fill remaining with heuristic if time runs out
            for t_rem in range(t, -1, -1):
                if thresholds[t_rem] == 0:
                    thresholds[t_rem] = K * 0.8
            break
            
        # Continuation value estimate for each path
        cont_val = disc * V[t+1]
        
        # Implicit threshold: Exercise if K - S_t >= cont_val
        # => S_t <= K - cont_val
        implicit_thresholds = K - cont_val
        
        # Robust threshold estimation: Median
        # This separates the "exercise" region from "continue" region in the training sample
        thresholds[t] = np.median(implicit_thresholds)
        
        # Update V[t] for the next step back
        # V[t] = max(Intrinsic, Continuation)
        intrinsic = np.maximum(K - S[t], 0.0)
        V[t] = np.maximum(intrinsic, cont_val)
        
    # Clip thresholds to reasonable range
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
