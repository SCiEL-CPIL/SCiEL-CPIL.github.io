"""Multi-objective optimizer on WFG7: Structure-aware search with efficient exploitation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search exploiting WFG7 structure with efficient local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        budget = self.budget
        dim = self.dim

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=np.float64))
            return f

        # Phase 1: Structured initialization exploiting WFG7 properties
        # WFG7's Pareto front is influenced by specific variable patterns.
        # Use a combination of uniform sampling and structured exploration.

        n_init = 30
        # Uniform random samples
        X_init = lo + span * rng.random((n_init, dim))
        for x in X_init:
            if used >= budget:
                break
            evaluate(x)

        # Add structured points along key dimensions
        for d in range(min(6, dim)):
            n_struct = max(2, (budget - used) // (6 * 4))
            n_struct = min(n_struct, 5)
            for _ in range(n_struct):
                if used >= budget or time.time() > deadline:
                    break
                x = mid.copy()
                x[d] = lo[d] + rng.random() * span[d]
                # Small perturbation to other dims
                x += 0.02 * span * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Phase 2: Efficient local exploitation around non-dominated points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)

            if len(non_dom_idx) == 0:
                sums = np.sum(F_arr, axis=1)
                non_dom_idx = np.argsort(sums)[:min(5, len(F_arr))]

            # Sort by sum of objectives to prioritize better points
            non_dom_idx = non_dom_idx[np.argsort(np.sum(F_arr[non_dom_idx], axis=1))]

            n_exploit = min(len(non_dom_idx), 8)
            for idx in non_dom_idx[:n_exploit]:
                if used >= budget or time.time() > deadline:
                    break

                center_x = all_X[idx].copy()
                center_f = all_F[idx].copy()
                remaining = budget - used
                n_steps = max(1, min(remaining // 4, 15))

                # Adaptive local search
                step_size = 0.08
                for _ in range(n_steps):
                    if used >= budget or time.time() > deadline:
                        break

                    best_trial = None
                    best_trial_f = None

                    # Try 2 random directions per step
                    for _ in range(2):
                        if used >= budget:
                            break
                        perturbation = step_size * span * rng.standard_normal(dim)
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1
                        all_X.append(trial_x.copy())
                        all_F.append(np.array(trial_f, dtype=np.float64))

                        if best_trial is None:
                            best_trial = trial_x
                            best_trial_f = trial_f
                        elif np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                            best_trial = trial_x
                            best_trial_f = trial_f
                        elif np.all(best_trial_f <= trial_f) and np.any(best_trial_f < trial_f):
                            pass  # Keep current best
                        else:
                            # Compare sums as tiebreaker
                            if np.sum(trial_f) < np.sum(best_trial_f):
                                best_trial = trial_x
                                best_trial_f = trial_f

                    if best_trial is not None:
                        if np.all(best_trial_f <= center_f) and np.any(best_trial_f < center_f):
                            center_x = best_trial
                            center_f = best_trial_f
                            step_size *= 0.8  # Shrink on improvement
                        else:
                            step_size *= 0.5  # Shrink more on no improvement
                # End steps for this center

        # Phase 3: Final refinement with very small steps
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            if len(non_dom_idx) > 0:
                non_dom_idx = non_dom_idx[np.argsort(np.sum(F_arr[non_dom_idx], axis=1))]
                for idx in non_dom_idx[:min(4, len(non_dom_idx))]:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.02
                    for _ in range(5):
                        if used >= budget or time.time() > deadline:
                            break
                        perturbation = step_size * span * rng.standard_normal(dim)
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1
                        all_X.append(trial_x.copy())
                        all_F.append(np.array(trial_f, dtype=np.float64))
                        if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                            center_x = trial_x
                            center_f = trial_f
                        step_size *= 0.7

        # Return all evaluated points (evaluator records all)
        if len(all_F) > 0:
            return np.array(all_X)
        return np.array([])

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points using vectorized comparison."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        # Vectorized dominance check
        # F[i] is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
        is_dominated = np.zeros(n, dtype=bool)

        # For each point, check if any other point dominates it
        for i in range(n):
            # F[j] <= F[i] for all objectives
            leq = np.all(F <= F[i], axis=1)
            # F[j] != F[i] (at least one strict)
            strict = np.any(F < F[i], axis=1)
            # Exclude self
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
