"""Multi-objective optimizer on WFG7: HV-aware exploitation with adaptive archive."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search with hypervolume-aware parent selection and adaptive exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 275.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([3.0, 5.0, 7.0])  # Reference point for hypervolume

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

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

        # Phase 1: Compact initialization - fewer points, more budget for exploitation
        n_init = 20
        X_init = lo + span * rng.random((n_init, dim))
        for x in X_init:
            if used >= budget or time.time() > deadline:
                break
            evaluate(x)

        # Phase 2: HV-aware exploitation loop
        # Repeatedly select parent by hypervolume contribution, do local search, update archive
        max_exploit_rounds = 40
        for round_num in range(max_exploit_rounds):
            if used >= budget or time.time() > deadline:
                break

            F_arr = np.array(all_F)
            n_points = len(F_arr)
            if n_points < 2:
                break

            # Get non-dominated set
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            if len(non_dom_idx) == 0:
                sums = np.sum(F_arr, axis=1)
                non_dom_idx = np.argsort(sums)[:min(3, n_points)]

            # Compute hypervolume contribution for each non-dominated point
            # Approximate: contribution = volume of region where this point is the "best"
            # Use a simple proxy: distance to reference point weighted by uniqueness
            nd_F = F_arr[non_dom_idx]
            # Filter points better than reference in all objectives
            valid_mask = np.all(nd_F < ref, axis=1)
            if not np.any(valid_mask):
                # Fallback: use all non-dominated
                valid_nd_idx = non_dom_idx
                valid_nd_F = nd_F
            else:
                valid_nd_idx = non_dom_idx[valid_mask]
                valid_nd_F = nd_F[valid_mask]

            if len(valid_nd_idx) == 0:
                valid_nd_idx = non_dom_idx
                valid_nd_F = nd_F

            # Score each candidate by a hypervolume proxy:
            # - Closer to reference = potentially higher contribution
            # - More "spread out" from other points = higher contribution
            if len(valid_nd_F) > 1:
                # Distance to reference (normalized)
                dist_ref = np.sqrt(np.sum((valid_nd_F - ref) ** 2, axis=1))
                dist_ref_norm = dist_ref / (np.max(dist_ref) + 1e-10)

                # Minimum distance to other non-dominated points (diversity)
                min_dists = np.zeros(len(valid_nd_F))
                for i in range(len(valid_nd_F)):
                    if i == 0:
                        continue
                    d = np.sqrt(np.sum((valid_nd_F[i] - valid_nd_F[:i]) ** 2, axis=1))
                    min_dists[i] = np.min(d) if len(d) > 0 else 0
                min_dists_norm = min_dists / (np.max(min_dists) + 1e-10)

                # Combined score: prefer points that are close to ref AND diverse
                scores = 0.6 * (1.0 - dist_ref_norm) + 0.4 * min_dists_norm
            else:
                scores = np.array([1.0])

            # Select top candidates for exploitation
            n_parents = min(len(valid_nd_idx), max(2, (budget - used) // 20))
            parent_indices = valid_nd_idx[np.argsort(-scores)[:n_parents]]

            for p_idx in parent_indices:
                if used >= budget or time.time() > deadline:
                    break

                center_x = all_X[p_idx].copy()
                center_f = all_F[p_idx].copy()
                remaining = budget - used

                # Adaptive step size based on recent progress
                step_size = 0.05
                n_steps = max(1, min(remaining // 5, 12))

                for step in range(n_steps):
                    if used >= budget or time.time() > deadline:
                        break

                    best_trial = None
                    best_trial_f = None
                    n_trials = 3

                    for t in range(n_trials):
                        if used >= budget:
                            break
                        # Mix of random and directed perturbations
                        if t == 0:
                            # Directed: toward reference point in objective space
                            # Approximate gradient direction by moving toward center of bounds
                            direction = -0.3 * rng.standard_normal(dim)
                        elif t == 1:
                            # Moderate random
                            direction = 0.1 * rng.standard_normal(dim)
                        else:
                            # Large random for exploration
                            direction = 0.15 * rng.standard_normal(dim)

                        perturbation = step_size * span * direction
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = np.array(problem(trial_x), dtype=np.float64)
                        used += 1
                        all_X.append(trial_x.copy())
                        all_F.append(trial_f.copy())

                        if best_trial is None:
                            best_trial = trial_x
                            best_trial_f = trial_f
                        elif np.all(trial_f <= best_trial_f) and np.any(trial_f < best_trial_f):
                            best_trial = trial_x
                            best_trial_f = trial_f
                        elif np.sum(trial_f) < np.sum(best_trial_f):
                            best_trial = trial_x
                            best_trial_f = trial_f

                    if best_trial is not None:
                        if np.all(best_trial_f <= center_f) and np.any(best_trial_f < center_f):
                            center_x = best_trial
                            center_f = best_trial_f
                            step_size *= 0.85  # Shrink moderately on improvement
                        else:
                            step_size *= 0.6  # Shrink more on stagnation

        # Phase 3: Final polish - very small steps around best points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            if len(non_dom_idx) > 0:
                # Prioritize by sum (simple proxy for quality)
                non_dom_idx = non_dom_idx[np.argsort(np.sum(F_arr[non_dom_idx], axis=1))]
                for idx in non_dom_idx[:min(3, len(non_dom_idx))]:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.01
                    for _ in range(8):
                        if used >= budget or time.time() > deadline:
                            break
                        perturbation = step_size * span * rng.standard_normal(dim)
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = np.array(problem(trial_x), dtype=np.float64)
                        used += 1
                        all_X.append(trial_x.copy())
                        all_F.append(trial_f.copy())
                        if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                            center_x = trial_x
                            center_f = trial_f
                        step_size *= 0.7

        if len(all_F) > 0:
            return np.array(all_X)
        return np.array([])

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            leq = np.all(F <= F[i], axis=1)
            strict = np.any(F < F[i], axis=1)
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
