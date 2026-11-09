"""Bermudan exercise-policy: Longstaff-Schwartz Monte Carlo method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps

    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]

    # Discount factors for each exercise date
    disc = np.exp(-r * np.arange(1, n_steps + 1) * dt)

    # Terminal payoff at maturity
    terminal_payoff = np.maximum(K - training_paths[:, -1], 0.0)

    # Feature function for regression
    def make_features(S):
        S = np.asarray(S, dtype=np.float64)
        S_clipped = np.clip(S, 1e-8, None)
        logS = np.log(S_clipped)
        # Polynomial basis in log-space
        return np.column_stack([
            np.ones_like(logS),
            logS,
            logS ** 2,
            logS ** 3,
            logS ** 4
        ])

    # Backward induction using LS regression
    V_next = terminal_payoff.copy()
    thresholds = {}

    for t in range(n_steps, 0, -1):
        if time.time() > deadline - 15.0:
            break

        S_t = training_paths[:, t - 1]
        exercise_payoff = np.maximum(K - S_t, 0.0)

        # Fit continuation value: E[V_next | S_t] via regression
        X = make_features(S_t)

        # Ridge regression
        reg_param = 1e-6 * np.trace(X.T @ X) / X.shape[1]
        A = X.T @ X + reg_param * np.eye(X.shape[1])
        b = X.T @ V_next

        try:
            coeffs = np.linalg.solve(A, b)
            cont_pred = X @ coeffs
            cont_value = disc[t - 1] * cont_pred
        except np.linalg.LinAlgError:
            cont_value = disc[t - 1] * np.mean(V_next) * np.ones(n_paths)

        # Exercise decision
        exercise_mask = exercise_payoff > cont_value

        # Determine threshold
        if np.any(exercise_mask) and not np.all(exercise_mask):
            sorted_idx = np.argsort(S_t)
            S_sorted = S_t[sorted_idx]
            ex_sorted = exercise_mask[sorted_idx]

            flip_idx = np.where(~ex_sorted)[0]
            if len(flip_idx) > 0:
                if flip_idx[0] > 0:
                    threshold = 0.5 * (S_sorted[flip_idx[0] - 1] + S_sorted[flip_idx[0]])
                else:
                    threshold = S_sorted[0]
            else:
                threshold = S_sorted[-1]
        elif np.all(exercise_mask):
            threshold = S_t.max() * 1.1
        else:
            threshold = S_t.min() * 0.9

        thresholds[t] = threshold

        # Update value for next iteration
        V_next = np.maximum(exercise_payoff, cont_value)

    class ExercisePolicy:
        def __init__(self, thresholds, n_steps, K):
            self.thresholds = thresholds
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            cs = np.asarray(current_state, dtype=np.float64)
            if time not in self.thresholds:
                return cs < self.K
            return cs < self.thresholds[time]

    return ExercisePolicy(thresholds, n_steps, K)
# EVOLVE-BLOCK-END
