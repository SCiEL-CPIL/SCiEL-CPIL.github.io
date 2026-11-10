import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 115.0  # 115s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    n = n_exercises
    dt = T / n

    # Convert training_paths to array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_paths, n_steps = paths.shape

    # Determine number of exercise dates from paths
    if n_steps > n_exercises:
        paths = paths[:, -n_exercises - 1:] if n_steps >= n_exercises + 1 else paths
    n_steps = paths.shape[1]
    n_ex = min(n_exercises, n_steps - 1) if n_steps > 1 else n_exercises
    n = n_ex
    dt = T / n

    # Precompute discount factors
    disc_factors = np.array([math.exp(-r * j * dt) for j in range(n + 1)])

    # Precompute intrinsic values for all paths and all dates
    intrinsic = np.maximum(K - paths[:, 1:n+1], 0.0)  # shape (n_paths, n)

    # --- Step 1: Initialize boundaries using a quick Monte Carlo estimate ---
    # For each date j, estimate the optimal boundary as the S value where
    # the path's intrinsic value exceeds its estimated continuation value.
    # We use a simple heuristic: for puts, boundary is typically below K.
    
    # Quick initialization: use quantile-based approach
    # For a Bermudan put, the exercise boundary decreases over time
    # (you're less likely to exercise early if the option is deep ITM)
    boundaries = np.zeros(n + 1)
    boundaries[n] = K  # At maturity, always exercise if ITM

    for j in range(n - 1, 0, -1):
        S_j = paths[:, j]
        # Estimate continuation: average of future intrinsic values weighted by discount
        # Simple approximation: continuation ≈ df * E[intrinsic at j+1 | S_j]
        # Use a rough estimate based on the current boundary for j+1
        if j + 1 <= n:
            # Paths that would continue past j
            cont_mask = S_j > boundaries[j + 1] if j + 1 <= n else np.ones(n_paths, dtype=bool)
            if np.sum(cont_mask) > 10:
                # Estimate continuation as discounted expected future payoff
                future_intrinsic = intrinsic[:, j:]  # from j to n
                # Approximate: continuation ≈ df * mean of intrinsic at j+1 for continuing paths
                V_cont_est = disc_factors[j+1] * np.mean(intrinsic[:, j] * cont_mask) / max(np.sum(cont_mask), 1)
            else:
                V_cont_est = 0.0
        else:
            V_cont_est = 0.0

        # Find boundary where intrinsic ≈ continuation estimate
        # For paths with S_j < K, intrinsic = K - S_j
        # We want to find S* such that K - S* ≈ V_cont_est
        # So S* ≈ K - V_cont_est
        if V_cont_est > 0:
            boundaries[j] = max(1.0, K - V_cont_est)
        else:
            boundaries[j] = K * 0.95

    # Ensure boundaries are non-decreasing as we go back in time
    # (you should be more willing to exercise when further from maturity)
    for j in range(n - 2, -1, -1):
        if j + 1 <= n:
            boundaries[j] = max(boundaries[j], boundaries[j + 1])

    # --- Step 2: Stochastic gradient ascent optimization ---
    # We'll use minibatch SGD to maximize expected discounted payoff
    # The policy is: exercise at date j if S_j <= boundaries[j]

    def compute_payoff(bnds, path_subset=None):
        """Compute total discounted payoff for given boundaries and path subset."""
        if path_subset is not None:
            S_subset = paths[:, 1:n+1][path_subset]
            intrinsic_subset = intrinsic[path_subset]
        else:
            S_subset = paths[:, 1:n+1]
            intrinsic_subset = intrinsic

        total_payoff = np.zeros(len(intrinsic_subset), dtype=np.float64)
        # Track which paths have already been exercised
        exercised = np.zeros(len(intrinsic_subset), dtype=bool)

        for j in range(n):
            # Only consider paths not yet exercised
            active = ~exercised
            if not np.any(active):
                break

            # Exercise condition: S_j <= boundary
            exercise_mask = S_subset[:, j] <= bnds[j + 1]
            exercise_mask = exercise_mask & active

            # Add payoff for exercised paths
            if np.any(exercise_mask):
                total_payoff[exercise_mask] += disc_factors[j + 1] * intrinsic_subset[exercise_mask, j]
                exercised[exercise_mask] = True

        return np.mean(total_payoff)

    def compute_gradient(bnds, batch_size=512):
        """Compute gradient of payoff w.r.t. boundaries using finite differences."""
        grad = np.zeros(n)
        eps = 0.5

        for j in range(n - 1):  # Don't optimize maturity boundary
            # Forward difference
            bnds_plus = bnds.copy()
            bnds_plus[j + 1] = bnds[j + 1] + eps
            payoff_plus = compute_payoff(bnds_plus)

            # Backward difference
            bnds_minus = bnds.copy()
            bnds_minus[j + 1] = max(1.0, bnds[j + 1] - eps)
            payoff_minus = compute_payoff(bnds_minus)

            if bnds[j + 1] + eps > bnds[j + 1]:
                grad[j] = (payoff_plus - payoff_minus) / (bnds_plus[j + 1] - bnds_minus[j + 1])
            else:
                grad[j] = 0.0

        return grad

    # Main optimization loop with adaptive step size
    best_bnds = boundaries.copy()
    best_payoff = compute_payoff(best_bnds)

    # Use full batch for stability, then switch to minibatch
    n_iter_full = 20
    n_iter_mini = 50
    batch_size = 1024

    for it in range(n_iter_full + n_iter_mini):
        if time.time() > deadline - 10.0:
            break

        if it < n_iter_full:
            # Full batch for stability
            grad = compute_gradient(best_bnds)
            lr = 1.0 / (it + 10)
        else:
            # Minibatch for speed
            batch_idx = np.random.choice(n_paths, batch_size, replace=False)
            # Compute gradient on subset
            grad = np.zeros(n)
            eps = 0.5
            for j in range(n - 1):
                bnds_plus = best_bnds.copy()
                bnds_plus[j + 1] = best_bnds[j + 1] + eps
                payoff_plus = compute_payoff(bnds_plus, batch_idx)

                bnds_minus = best_bnds.copy()
                bnds_minus[j + 1] = max(1.0, best_bnds[j + 1] - eps)
                payoff_minus = compute_payoff(bnds_minus, batch_idx)

                if best_bnds[j + 1] + eps > best_bnds[j + 1]:
                    grad[j] = (payoff_plus - payoff_minus) / (bnds_plus[j + 1] - bnds_minus[j + 1])
                else:
                    grad[j] = 0.0
            lr = 0.5 / (it - n_iter_full + 10)

        # Update boundaries
        for j in range(n - 1):
            if grad[j] > 0:
                best_bnds[j + 1] = min(K, best_bnds[j + 1] + lr * grad[j])
            elif grad[j] < 0:
                best_bnds[j + 1] = max(1.0, best_bnds[j + 1] + lr * grad[j])

        # Ensure boundaries are valid
        for j in range(1, n + 1):
            best_bnds[j] = np.clip(best_bnds[j], 1.0, K)

        # Evaluate and keep best
        current_payoff = compute_payoff(best_bnds)
        if current_payoff > best_payoff:
            best_payoff = current_payoff
        else:
            # Revert if worse
            best_bnds = boundaries.copy()

    # --- Step 3: Final coordinate ascent refinement ---
    for iteration in range(10):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for j in range(1, n):
            current = best_bnds[j]
            for step_frac in [0.02, 0.01, 0.005]:
                step = max(0.2, current * step_frac)
                for delta in [-step, step]:
                    trial = best_bnds.copy()
                    trial[j] = max(1.0, min(K, current + delta))
                    payoff = compute_payoff(trial)
                    if payoff > best_payoff + 1e-9:
                        best_bnds[j] = trial[j]
                        best_payoff = payoff
                        improved = True
        if not improved:
            break

    boundaries = best_bnds

    # Final fallback
    for j in range(1, n + 1):
        if boundaries[j] <= 0:
            boundaries[j] = K * 0.95

    class ExercisePolicy:
        def __init__(self, bnds, K_val):
            self.boundaries = bnds
            self.K = K_val

        def exercise(self, current_state, time_param):
            S_arr = np.asarray(current_state, dtype=np.float64)

            if isinstance(time_param, (int, float, np.integer, np.floating)):
                t = int(time_param)
                if t < 1:
                    t = 1
                elif t > 12:
                    t = 12
                return S_arr <= self.boundaries[t]
            else:
                t_arr = np.asarray(time_param)
                t_clipped = np.clip(t_arr, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in t_clipped])
                if S_arr.ndim == 1 and b.ndim == 0:
                    return S_arr <= float(b)
                elif S_arr.ndim == 1 and b.ndim == 1:
                    return S_arr <= b
                else:
                    return S_arr <= b

    return ExercisePolicy(boundaries, K)
