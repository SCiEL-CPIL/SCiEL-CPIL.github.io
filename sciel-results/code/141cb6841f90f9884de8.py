import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)

    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}

    # Pass 1: Compute approximate boundaries with a log-uniform grid
    # Log-uniform gives better resolution at low S where put boundaries are sensitive
    S_min_1 = 0.01
    S_max_1 = 1.5 * K
    n_grid_1 = 20000
    S_grid_1 = np.geomspace(S_min_1, S_max_1, n_grid_1)

    V_next_1 = np.maximum(K - S_grid_1, 0.0)
    boundaries_approx = np.zeros(n_exercises + 1)

    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid_1, 0.0)
        Su = S_grid_1 * u
        Sd = S_grid_1 * d

        Su_clipped = np.clip(Su, S_grid_1[0], S_grid_1[-1])
        Sd_clipped = np.clip(Sd, S_grid_1[0], S_grid_1[-1])

        V_up = np.interp(Su_clipped, S_grid_1, V_next_1)
        V_down = np.interp(Sd_clipped, S_grid_1, V_next_1)

        continuation = df * (p * V_up + (1 - p) * V_down)
        V_current = np.maximum(intrinsic, continuation)

        if step in exercise_steps:
            idx = exercise_steps[step]
            diff = intrinsic - continuation
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                if last_idx < n_grid_1 - 1:
                    d0 = diff[last_idx]
                    d1 = diff[last_idx + 1]
                    s0 = S_grid_1[last_idx]
                    s1 = S_grid_1[last_idx + 1]
                    if d0 != d1:
                        t = d0 / (d0 - d1)
                        boundaries_approx[idx] = s0 + t * (s1 - s0)
                    else:
                        boundaries_approx[idx] = s0
                else:
                    boundaries_approx[idx] = S_grid_1[-1]
            else:
                boundaries_approx[idx] = 0.0

        V_next_1 = V_current

    boundaries_approx[12] = K

    # Check time - if we're running low, just return with approximate boundaries
    if time.time() > deadline - 30:
        class ExercisePolicy:
            def __init__(self, boundaries, K):
                self.boundaries = boundaries
                self.K = K

            def exercise(self, current_state, time):
                if isinstance(time, (int, float, np.integer, np.floating)):
                    idx = int(time)
                    idx = max(1, min(12, idx))
                    b = self.boundaries[idx]
                    return np.array(current_state) <= b
                else:
                    times = np.array(time)
                    times_clipped = np.clip(times, 1, 12).astype(int)
                    b = np.array([self.boundaries[t] for t in times_clipped])
                    return np.array(current_state) <= b

        return ExercisePolicy(boundaries_approx, K)

    # Pass 2: Refine each boundary with a very fine uniform grid focused on the boundary region
    # For each exercise date, create a fine grid centered around the approximate boundary
    boundaries_refined = np.zeros(n_exercises + 1)
    boundaries_refined[12] = K

    for idx in range(1, 12):
        b_approx = boundaries_approx[idx]
        if b_approx <= 0:
            boundaries_refined[idx] = 0.0
            continue

        # Create a fine grid around the approximate boundary
        # Range: [0.5*b_approx, 1.5*b_approx] to capture the boundary region
        margin = 0.5 * b_approx
        S_min_2 = max(0.001, b_approx - margin)
        S_max_2 = min(1.5 * K, b_approx + margin)
        n_grid_2 = 50000
        S_grid_2 = np.linspace(S_min_2, S_max_2, n_grid_2)

        # Recompute the option value at this specific exercise date with the fine grid
        # We need to run backward induction from maturity to this exercise date
        # But that's expensive. Instead, we'll use the fact that the boundary is determined
        # by where intrinsic = continuation. We can evaluate the continuation value
        # at the fine grid points using the already-computed V_next from pass 1.

        # Actually, let's do a proper backward induction but only for the steps
        # from maturity back to this exercise date, using the fine grid.
        # This is more accurate but requires care.

        # Alternative: Use the pass 1 result as a starting point and do a local refinement.
        # The boundary is where diff(S) = intrinsic(S) - continuation(S) = 0.
        # We can evaluate diff on the fine grid and find the zero crossing.

        # To get continuation values on the fine grid, we need V_next at Su and Sd.
        # We can interpolate V_next_1 (from pass 1) at the fine grid points.

        # But wait - V_next_1 was computed on S_grid_1. We need to be careful.
        # Let's just do a full backward induction on the fine grid for this specific date.
        # It's 12 dates * (n_steps - exercise_step) steps * n_grid_2 operations.
        # For date 1: 1100 steps * 50000 = 55M ops. That might be too slow for 11 dates.

        # Better approach: Use the pass 1 V_next at the exercise step as a reference.
        # The continuation value at the exercise date depends on future values.
        # We can approximate by interpolating the pass 1 results.

        # Let's try a simpler refinement: evaluate the diff function on the fine grid
        # using interpolated continuation values from pass 1.

        # Get the step index for this exercise date
        step_idx = exercise_steps[idx]

        # We need V_next at step_idx (i.e., values one step after the exercise date)
        # From pass 1, we don't have stored V at each step, only the final result.
        # Let's recompute backward induction but store V at the exercise step.

        # This is getting complex. Let's use a different approach:
        # Do a second full backward induction but with a hybrid grid:
        # Use the log-uniform grid for most of the range, but add extra points
        # around each approximate boundary.

        # Actually, the simplest effective approach: do a full backward induction
        # with a very fine uniform grid over [0, 1.2*K] which covers all boundaries.
        # 100k points on [0, 120] gives 0.0012 resolution. That's very fine.

        # But we already did that in the original. Let's try a different refinement:
        # Use the approximate boundaries to create a non-uniform grid with
        # extra density around each boundary.

        # For now, let's just use a fine uniform grid over [0, 1.2*K] with 100k points
        # and do a clean backward induction. This is the same as the original but
        # we're doing it as a "refinement" pass.

        pass  # Skip for now, use pass 1 boundaries

    # Actually, let's just use the pass 1 boundaries but with a better grid.
    # Redo the entire computation with a uniform grid of 100k points on [0, 1.5*K]
    # This is what the original did. The log-uniform grid in pass 1 might have
    # introduced errors. Let's combine: use log-uniform for the main computation
    # but verify boundaries are reasonable.

    # Final decision: Use the log-uniform grid result from pass 1.
    # It should be more accurate than uniform for put options because
    # the boundary is at low S where log-uniform has higher density.
    boundaries_final = boundaries_approx.copy()
    boundaries_final[12] = K

    # Clamp boundaries to reasonable range [0, K]
    boundaries_final = np.clip(boundaries_final, 0.0, K)

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K

        def exercise(self, current_state, time):
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b

    return ExercisePolicy(boundaries_final, K)
