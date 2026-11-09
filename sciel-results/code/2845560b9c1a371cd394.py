"""Multi-objective optimizer for RE37 using Sobol sweep + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol quasi-random sweep followed by local Gaussian refinement around non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_non_dominated(self, F):
        """Return boolean mask: True for points not dominated by any other point in F."""
        n = len(F)
        if n == 0:
            return np.array([])
        # F[i] is dominated by F[j] if F[j] <= F[i] in all objectives and F[j] < F[i] in at least one
        # Vectorized: for each pair (i, j), check domination
        # Better: for each i, check if any j dominates i
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            # F[j] dominates F[i] if all(F[j] <= F[i]) and any(F[j] < F[i])
            le = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            # Exclude self
            le[i] = False
            if np.any(le & lt):
                dominated[i] = True
        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Sobol quasi-random sweep
        # Use a Sobol sequence for uniform coverage
        # We want to use about 60% of budget for the sweep, 40% for local refinement
        sweep_count = int(budget * 0.6)
        local_count = budget - sweep_count

        # Generate Sobol points in [0,1]^dim, then map to bounds
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Need a power of 2 for Sobol; use next power of 2 >= sweep_count
        n_sobol = 1
        while n_sobol < sweep_count:
            n_sobol *= 2
        # Generate n_sobol points, take first sweep_count
        sobol_unit = sampler.random(n_sobol)
        sobol_unit = sobol_unit[:sweep_count]
        X_sobol = lo + (hi - lo) * sobol_unit

        # Evaluate all Sobol points
        F_all = np.zeros((sweep_count, n_obj))
        for i in range(sweep_count):
            F_all[i] = problem(X_sobol[i])

        used = sweep_count

        # Find non-dominated points from the sweep
        nd_mask = self._is_non_dominated(F_all)
        nd_indices = np.where(nd_mask)[0]
        X_nd = X_sobol[nd_indices]
        F_nd = F_all[nd_indices]

        # Phase 2: Local Gaussian refinement around non-dominated points
        # For each non-dominated point, generate local perturbations
        # Use a shrinking step size
        rng = np.random.default_rng(42)
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]

        # We'll do local refinement: for each local budget slot, pick a random ND point
        # and perturb it
        if len(X_nd) > 0 and local_count > 0:
            # Sort ND points by some criterion to prioritize
            # Use sum of objectives as a simple ranking
            sums = np.sum(F_nd, axis=1)
            order = np.argsort(sums)
            X_nd_sorted = X_nd[order]
            F_nd_sorted = F_nd[order]

            # For each local evaluation, pick a point from the ND set with probability
            # proportional to 1/(rank+1) to favor better points but maintain diversity
            for k in range(local_count):
                if time.time() > deadline:
                    break

                # Pick which step size to use based on progress
                progress = k / max(local_count, 1)
                if progress < 0.2:
                    step = step_sizes[0]
                elif progress < 0.4:
                    step = step_sizes[1]
                elif progress < 0.6:
                    step = step_sizes[2]
                elif progress < 0.8:
                    step = step_sizes[3]
                else:
                    step = step_sizes[4]

                # Pick a parent from ND points, weighted toward better ones
                n_nd = len(X_nd_sorted)
                # Inverse rank weighting
                weights = 1.0 / (np.arange(n_nd) + 1)
                weights /= weights.sum()
                parent_idx = rng.choice(n_nd, p=weights)
                parent = X_nd_sorted[parent_idx]

                # Generate a perturbation
                child = parent + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Add to ND set if it's non-dominated w.r.t. current ND set
                # Check if child is dominated by any ND point
                if len(F_nd_sorted) > 0:
                    le = np.all(F_nd_sorted <= fc, axis=1)
                    lt = np.any(F_nd_sorted < fc, axis=1)
                    if np.any(le & lt):
                        # Child is dominated, skip
                        continue

                # Check if child dominates any existing ND points
                if len(F_nd_sorted) > 0:
                    le2 = np.all(fc <= F_nd_sorted, axis=1)
                    lt2 = np.any(fc < F_nd_sorted, axis=1)
                    dominated_idx = np.where(le2 & lt2)[0]
                    if len(dominated_idx) > 0:
                        # Remove dominated points
                        mask = np.ones(len(F_nd_sorted), dtype=bool)
                        mask[dominated_idx] = False
                        X_nd_sorted = X_nd_sorted[mask]
                        F_nd_sorted = F_nd_sorted[mask]

                # Add child to ND set
                X_nd_sorted = np.vstack([X_nd_sorted, child.reshape(1, -1)])
                F_nd_sorted = np.vstack([F_nd_sorted, fc.reshape(1, -1)])

        return X_sobol
# EVOLVE-BLOCK-END
