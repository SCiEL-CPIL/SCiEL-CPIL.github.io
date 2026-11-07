"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Two-phase multi-objective optimizer: Sobol exploration + archive-based DE refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Phase 1: Sobol sequence for better low-discrepancy coverage
        sobol_size = min(128, self.budget // 2)
        sobol_n = 1
        while sobol_n < sobol_size:
            sobol_n *= 2
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        sobol_samples = sampler.random(n=sobol_n)
        X_sobol = lo + (hi - lo) * sobol_samples[:sobol_size]
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = sobol_size

        # Build initial archive from non-dominated points (vectorized)
        archive_idx = self._nondominated_indices(F_sobol)
        archive_X = X_sobol[archive_idx].copy()
        archive_F = F_sobol[archive_idx].copy()

        # Phase 2: Archive-based DE refinement
        pop_size = min(40, max(15, (self.budget - used) // 4))
        if len(archive_X) < pop_size:
            n_rand = pop_size - len(archive_X)
            X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_rand
            X_init = np.vstack([archive_X, X_rand])
            F_init = np.vstack([archive_F, F_rand])
        else:
            X_init = archive_X[:pop_size].copy()
            F_init = archive_F[:pop_size].copy()

        # DE parameters
        CR = 0.85
        F_scale = 0.7

        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Select parents from archive
            if len(archive_X) >= 3:
                p1_idx = rng.integers(0, len(archive_X))
                p2_idx = rng.integers(0, len(archive_X))
                p3_idx = rng.integers(0, len(archive_X))
                parent1 = archive_X[p1_idx]
                parent2 = archive_X[p2_idx]
                parent3 = archive_X[p3_idx]
            else:
                parent1 = X_init[rng.integers(0, len(X_init))]
                parent2 = X_init[rng.integers(0, len(X_init))]
                parent3 = X_init[rng.integers(0, len(X_init))]

            # DE mutation + crossover
            v = parent1 + F_scale * (parent2 - parent3)
            j_rand = rng.integers(0, self.dim)
            mask = rng.random(self.dim) < CR
            mask[j_rand] = True
            child = np.where(mask, v, parent1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Fast archive update
            if len(archive_F) > 0:
                child_dominated = np.any(
                    (archive_F <= fc).all(axis=1) & (archive_F < fc).any(axis=1)
                )
            else:
                child_dominated = False

            if not child_dominated:
                if len(archive_F) > 0:
                    dominated = (
                        (fc <= archive_F).all(axis=1) & (fc < archive_F).any(axis=1)
                    )
                    if dominated.any():
                        keep_mask = ~dominated
                        archive_X = archive_X[keep_mask]
                        archive_F = archive_F[keep_mask]

                archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

                if len(archive_X) > 150:
                    archive_X, archive_F = self._reduce_archive(archive_X, archive_F)

            # Update population
            if len(F_init) > 0:
                child_dom = (
                    (fc <= F_init).all(axis=1) & (fc < F_init).any(axis=1)
                )
                if child_dom.any():
                    dom_idx = np.where(child_dom)[0]
                    replace_idx = dom_idx[rng.integers(0, len(dom_idx))]
                else:
                    worst = int(np.argmax(np.sum(F_init, axis=1)))
                    replace_idx = worst
                X_init[replace_idx] = child
                F_init[replace_idx] = fc

        return X_init

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F (vectorized)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        # F[i] is dominated by F[j] if F[j] <= F[i] and F[j] < F[i] in at least one obj
        # Build domination matrix: dom[i,j] = True if j dominates i
        # F[j] <= F[i] for all objectives: (F[j] <= F[i]).all(axis=1)
        # F[j] < F[i] for at least one: (F[j] < F[i]).any(axis=1)
        # Compare all pairs: for each i, check all j
        # Shape: (n, n) where [i,j] = j dominates i
        leq = (F[:, np.newaxis, :] <= F[np.newaxis, :, :]).all(axis=2)  # (n,n): F[j]<=F[i]
        lt = (F[:, np.newaxis, :] < F[np.newaxis, :, :]).any(axis=2)     # (n,n): F[j]<F[i]
        dom_matrix = leq & lt  # dom_matrix[i,j] = j dominates i
        np.fill_diagonal(dom_matrix, False)
        dominated = dom_matrix.any(axis=1)  # True if any j dominates i
        return np.where(~dominated)[0]

    def _tournament_select(self, X, F, rng, k=3):
        """Tournament selection with domination awareness."""
        idx = rng.integers(0, len(X), size=k)
        F_cand = F[idx]
        n_cand = len(idx)

        # Find non-dominated candidates
        non_dom = np.zeros(n_cand, dtype=bool)
        for i in range(n_cand):
            is_dom = False
            for j in range(n_cand):
                if i != j:
                    if (F_cand[j] <= F_cand[i]).all() and (F_cand[j] < F_cand[i]).any():
                        is_dom = True
                        break
            non_dom[i] = not is_dom

        if non_dom.any():
            candidates = idx[non_dom]
            return X[candidates[rng.integers(0, len(candidates))]]
        else:
            sums = np.sum(F_cand, axis=1)
            return X[idx[np.argmin(sums)]]

    def _reduce_archive(self, X, F, max_size=150):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F

        n = len(F)
        crowding = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                denom = F[order[i+1], m] - F[order[i-1], m]
                if denom > 1e-10:
                    crowding[order[i]] += 1.0 / denom

        keep = np.argsort(crowding)[-max_size:]
        keep = np.sort(keep)
        return X[keep], F[keep]


# EVOLVE-BLOCK-END
