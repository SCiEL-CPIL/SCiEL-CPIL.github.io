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
        # Use next power of 2 for Sobol
        sobol_n = 1
        while sobol_n < sobol_size:
            sobol_n *= 2
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        sobol_samples = sampler.random(n=sobol_n)
        X_sobol = lo + (hi - lo) * sobol_samples[:sobol_size]
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = sobol_size

        # Build initial archive from non-dominated points
        archive_X, archive_F = self._build_archive(X_sobol, F_sobol)

        # Phase 2: Archive-based DE refinement
        pop_size = min(40, max(15, (self.budget - used) // 4))
        # Initialize population from archive + random
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

        # DE parameters - slightly more aggressive
        CR = 0.85
        F_scale = 0.7

        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Select parents from archive using domination-aware tournament
            if len(archive_X) < 3:
                # Not enough in archive, pick from population
                parent1 = self._tournament_select(X_init, F_init, rng)
                parent2 = self._tournament_select(X_init, F_init, rng)
                parent3 = self._tournament_select(X_init, F_init, rng)
            else:
                parent1 = self._tournament_select(archive_X, archive_F, rng)
                parent2 = self._tournament_select(archive_X, archive_F, rng)
                parent3 = self._tournament_select(archive_X, archive_F, rng)

            # DE/BI (binary) crossover with adaptive mutation
            v = parent1 + F_scale * (parent2 - parent3)
            j_rand = rng.integers(0, self.dim)
            mask = rng.random(self.dim) < CR
            mask[j_rand] = True
            child = np.where(mask, v, parent1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Efficient archive update using vectorized domination check
            if len(archive_F) > 0:
                # Check if child is dominated by any archive member
                child_dominated = np.any(
                    (archive_F <= fc).all(axis=1) & (archive_F < fc).any(axis=1)
                )
            else:
                child_dominated = False

            if not child_dominated:
                # Remove dominated archive members using vectorized ops
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

                # Limit archive size
                if len(archive_X) > 150:
                    archive_X, archive_F = self._reduce_archive(archive_X, archive_F)

            # Update population: replace dominated or worst individual
            if len(X_init) > 0:
                # Find if child dominates any population member
                if len(F_init) > 0:
                    child_dom = (
                        (fc <= F_init).all(axis=1) & (fc < F_init).any(axis=1)
                    )
                    if child_dom.any():
                        # Replace one dominated individual
                        dom_idx = np.where(child_dom)[0]
                        replace_idx = dom_idx[rng.integers(0, len(dom_idx))]
                    else:
                        # Replace worst by hypervolume contribution approximation
                        worst = int(np.argmax(np.sum(F_init, axis=1)))
                        replace_idx = worst
                else:
                    replace_idx = 0
                X_init[replace_idx] = child
                F_init[replace_idx] = fc

        return X_init

    def _build_archive(self, X, F):
        """Build initial archive from non-dominated points using vectorized ops."""
        n = len(X)
        if n == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)

        # Vectorized domination check
        dominated = np.zeros(n, dtype=bool)
        # For each point, check if any other point dominates it
        for i in range(n):
            if dominated[i]:
                continue
            # Check if any j dominates i
            doms_i = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
            doms_i[i] = False  # exclude self
            if doms_i.any():
                dominated[i] = True

        mask = ~dominated
        return X[mask], F[mask]

    def _tournament_select(self, X, F, rng, k=3):
        """Tournament selection with domination awareness."""
        idx = rng.integers(0, len(X), size=k)
        # Check for domination among candidates
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
            # Pick randomly from non-dominated candidates
            candidates = idx[non_dom]
            return X[candidates[rng.integers(0, len(candidates))]]
        else:
            # All dominated, pick by sum
            sums = np.sum(F_cand, axis=1)
            return X[idx[np.argmin(sums)]]

    def _reduce_archive(self, X, F, max_size=150):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F

        # Compute crowding distance
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

        # Remove points with lowest crowding distance
        keep = np.argsort(crowding)[-max_size:]
        keep = np.sort(keep)
        return X[keep], F[keep]


# EVOLVE-BLOCK-END
