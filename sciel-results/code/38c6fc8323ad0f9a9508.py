"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    tp = np.asarray(training_paths, dtype=np.float64)

    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            paths = tp
            n_paths, n_times = tp.shape
        elif tp.shape[1] == 4096:
            paths = tp.T
            n_times, n_paths = tp.shape
        else:
            paths = tp
            n_paths, n_times = tp.shape
    else:
        paths = tp
        n_paths, n_times = paths.shape

    dt = T / n_dates
    discount = np.exp(-r * dt)

    # Backward induction
    S_last = paths[:, n_dates]
    V = np.maximum(K - S_last, 0.0)

    decisions = [None] * (n_dates + 1)
    decisions[n_dates] = V > 0

    for i in range(n_dates - 1, 0, -1):
        if time.time() > deadline - 5:
            break

        S_i = paths[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        cont_disc = discount * V

        decisions[i] = immediate >= cont_disc
        V = np.maximum(immediate, cont_disc)

    # Extract thresholds: for each time step, the max S where we exercise
    thresholds = {}
    for i in range(1, n_dates + 1):
        if decisions[i] is not None:
            S_i = paths[:, i]
            exercised = decisions[i]
            if np.any(exercised):
                threshold = float(np.max(S_i[exercised]))
            else:
                threshold = 0.0
            thresholds[i] = threshold
        else:
            # Fallback: use K for later dates, lower for earlier
            thresholds[i] = K * (i / n_dates)

    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            i = int(round(time * 12))
            i = max(1, min(12, i))
            threshold = self.thresholds.get(i, self.K)
            return (cs <= threshold).astype(bool)

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
