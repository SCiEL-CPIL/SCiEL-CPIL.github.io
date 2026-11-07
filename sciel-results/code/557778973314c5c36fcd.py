"""Multi-objective optimizer on WFG7: Differential-evolution style search with archive."""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def _update_archive(self, archive, x, f, max_size=64):
        """Insert into archive if not dominated; remove dominated points."""
        # Check if f is dominated by any point in archive
        for af in archive:
            if np.all(af <= f) and np.any(af < f):
                return archive  # dominated, do not insert

        # Remove points in archive dominated by f
        new_archive = []
        for ax, af in archive:
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))

        new_archive.append((x, f))

        # Limit archive size by keeping most spread points
        if len(new_archive) > max_size:
            # Sort by first objective, keep a spread subset
            new_archive.sort(key=lambda t: t[1][0])
            keep = []
            step = max(1, len(new_archive) // max_size)
            for i in range(0, len(new_archive), step):
                keep.append(new_archive[i])
            if len(keep) < max_size:
                keep.append(new_archive[-1])
            new_archive = keep

        return new_archive

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for DE
        pop_size = min(30, max(10, budget // 7))

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of best non-dominated points
        archive = []
        for i in range(pop_size):
            archive = self._update_archive(archive, X[i], F[i])

        # DE parameters
        F_param = 0.8  # differential weight
        CR = 0.9       # crossover rate

        # Main loop: differential evolution with archive-guided replacement
        while used < budget:
            if time.time() > deadline:
                break

            # Select base vector: random individual from population
            r = rng.integers(pop_size)

            # Select two distinct others
            a = rng.integers(pop_size)
            while a == r:
                a = rng.integers(pop_size)
            b = rng.integers(pop_size)
            while b == r or b == a:
                b = rng.integers(pop_size)

            # Create trial vector using DE/rand/1
            jrand = rng.integers(dim)
            v = X[r] + F_param * (X[a] - X[b])

            # Binomial crossover
            u = X[r].copy()
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    u[j] = v[j]

            # Clip to bounds
            u = np.clip(u, lo, hi)

            # Evaluate trial
            fu = problem(u)
            used += 1

            # Update archive
            archive = self._update_archive(archive, u, fu)

            # Replacement: accept if u dominates X[r], or use fitness-based
            # Use a simple dominance check
            if np.all(fu <= F[r]) and np.any(fu < F[r]):
                # u dominates r: replace
                X[r] = u
                F[r] = fu
            elif np.all(F[r] <= fu) and np.any(F[r] < fu):
                # r dominates u: reject
                pass
            else:
                # Non-dominated: accept if archive suggests improvement
                # Simple heuristic: accept with some probability based on progress
                progress = used / budget
                if rng.random() < 0.3 + 0.4 * progress:
                    X[r] = u
                    F[r] = fu

            # Occasionally inject archive points into population to maintain diversity
            if used % 50 == 0 and len(archive) > 0:
                # Pick a random archive point and place it in a random population slot
                ax, af = archive[rng.integers(len(archive))]
                slot = rng.integers(pop_size)
                # Only replace if archive point is non-dominated by the slot point
                if not (np.all(F[slot] <= af) and np.any(F[slot] < af)):
                    X[slot] = ax
                    F[slot] = af

        # Final polish: local search around best archive points
        if used < budget and len(archive) > 0:
            remaining = budget - used
            # Sort archive by sum of objectives (prefer lower)
            archive.sort(key=lambda t: np.sum(t[1]))
            n_polish = min(len(archive), max(1, remaining // 5))
            polish_each = max(1, remaining // n_polish)

            for idx in range(n_polish):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                best_x = archive[idx][0].copy()
                best_f = archive[idx][1].copy()
                best_sum = float(np.sum(best_f))

                # Coordinate-wise local search
                scale = 0.02 * span
                for step in range(polish_each):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try small perturbations in random directions
                    for _ in range(3):
                        if used >= budget:
                            break
                        direction = rng.standard_normal(dim)
                        direction /= (np.linalg.norm(direction) + 1e-12)
                        trial_x = np.clip(best_x + 0.01 * scale * direction, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1

                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x
                            best_f = trial_f
                            best_sum = float(np.sum(best_f))
                        elif float(np.sum(trial_f)) < best_sum - 1e-6:
                            # Accept if scalarized improvement
                            if rng.random() < 0.5:
                                best_x = trial_x
                                best_f = trial_f
                                best_sum = float(np.sum(best_f))

                # Update archive with polished point
                archive = self._update_archive(archive, best_x, best_f)

        # Return the population (the evaluator records all objective vectors from calls)
        return X
