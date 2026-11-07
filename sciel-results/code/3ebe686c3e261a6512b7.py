"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Hierarchical Sobol sampling + local refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Phase 1: Hierarchical Sobol sampling for broad, structured coverage
        # Use a Sobol sequence with increasing resolution
        n_sobol = min(self.budget - 40, 160)  # Reserve 40 for local refinement
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n=n_sobol)
        X_sobol = lo + (hi - lo) * sobol_samples
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        # Build initial archive from non-dominated points
        archive_X, archive_F = self._build_archive(X_sobol, F_sobol)

        # Phase 2: Local refinement around promising archive points
        # Select top-k archive points by hypervolume contribution and do local search
        n_refine = self.budget - used
        if n_refine > 0 and len(archive_X) > 0:
            # Select diverse promising points from archive
            # Use crowding distance to select diverse points
            if len(archive_X) > 10:
                keep_idx = self._select_diverse(archive_X, archive_F, k=min(10, len(archive_X)))
                center_X = archive_X[keep_idx]
                center_F = archive_F[keep_idx]
            else:
                center_X = archive_X
                center_F = archive_F

            # Local search around each center
            n_per_center = max(1, n_refine // len(center_X))
            rng = np.random.default_rng(123)
            step_sizes = [0.1, 0.05, 0.02, 0.01]  # Multi-scale steps

            for ci in range(len(center_X)):
                if used >= self.budget:
                    break
                if time.time() - start_time > deadline:
                    break

                for step_frac in step_sizes:
                    if used >= self.budget:
                        break
                    n_steps = max(1, n_per_center // len(step_sizes))
                    for _ in range(n_steps):
                        if used >= self.budget:
                            break
                        if time.time() - start_time > deadline:
                            break

                        # Generate perturbation
                        noise = rng.standard_normal(self.dim)
                        child = center_X[ci] + step_frac * (hi - lo) * noise
                        child = np.clip(child, lo, hi)

                        fc = problem(child)
                        used += 1

                        # Update archive
                        if len(archive_F) == 0:
                            archive_X = child.reshape(1, -1).copy()
                            archive_F = fc.reshape(1, -1).copy()
                        else:
                            # Check if child is dominated
                            child_dominated = np.any(
                                np.all(archive_F <= fc, axis=1) & np.any(archive_F < fc, axis=1)
                            )
                            if not child_dominated:
                                # Remove dominated archive members
                                dominated = np.array([
                                    np.all(fc <= f, axis=0) and np.any(fc < f)
                                    for f in archive_F
                                ])
                                if np.any(dominated):
                                    archive_X = np.delete(archive_X, np.where(dominated)[0], axis=0)
                                    archive_F = np.delete(archive_F, np.where(dominated)[0], axis=0)
                                archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                                archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

                                if len(archive_X) > 300:
                                    archive_X, archive_F = self._reduce_archive(archive_X, archive_F)

                        # Move center to child if it's better (for exploitation)
                        if len(archive_F) > 0:
                            # Check if fc is better than center_F[ci]
                            if np.all(fc <= center_F[ci]) and np.any(fc < center_F[ci]):
                                center_X[ci] = child
                                center_F[ci] = fc

        return archive_X if len(archive_X) > 0 else X_sobol[:1]

    def _build_archive(self, X, F):
        """Build initial archive from non-dominated points."""
        n = len(X)
        if n == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)

        # Vectorized dominance check
        # F[i] dominates F[j] if F[i] <= F[j] and F[i] < F[j] in at least one obj
        # For each i, check if any j dominates i
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            # Check if any other point dominates i
            dom_mask = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dom_mask[i] = False  # Don't consider self
            if np.any(dom_mask):
                dominated[i] = True

        mask = ~dominated
        return X[mask], F[mask]

    def _select_diverse(self, X, F, k=10):
        """Select k diverse points from archive using crowding distance."""
        n = len(F)
        if n <= k:
            return np.arange(n)

        # Compute crowding distance
        crowding = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                if F[order[i+1], m] - F[order[i-1], m] > 1e-10:
                    crowding[order[i]] += (F[order[i+1], m] - F[order[i-1], m])

        # Select top-k by crowding distance
        keep = np.argsort(crowding)[-k:]
        return np.sort(keep)

    def _reduce_archive(self, X, F, max_size=300):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F

        keep_idx = self._select_diverse(X, F, k=max_size)
        return X[keep_idx], F[keep_idx]


# EVOLVE-BLOCK-END
