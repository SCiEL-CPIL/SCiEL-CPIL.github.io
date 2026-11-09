"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution-style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use global numpy random (seeds 0 and 1 are set by evaluator)
        rng = np.random

        # Initial population: random + a few structured points
        pop_size = min(40, max(10, budget // 5))
        X = lo + span * rng.random((pop_size, dim))

        # Add some structured points to help explore
        structured = np.array([
            lo,
            hi,
            (lo + hi) / 2,
            lo + 0.25 * span,
            lo + 0.75 * span,
        ])
        # Add corner points
        corners = []
        for i in range(2 ** dim):
            corner = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            corners.append(corner)
        structured = np.vstack([structured, np.array(corners)])
        # Take up to pop_size from structured, filling in with random
        n_struct = min(len(structured), pop_size)
        X = np.vstack([structured[:n_struct], X[n_struct:]])
        # Trim to pop_size
        X = X[:pop_size]

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Maintain an archive of non-dominated solutions
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        def is_dominated(f, archive_F_list):
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, af):
            return np.all(f <= af) and np.any(f < af)

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if new point is dominated by archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, skip
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Add initial points to archive
        archive_X = []
        archive_F = []
        for i in range(len(X)):
            update_archive(X[i], F[i])

        # Main loop: differential evolution with mutation
        F_current = F.copy()
        X_current = X.copy()

        # Mutation parameters
        CR = 0.7  # crossover rate
        F_mut = 0.5  # mutation scale

        while used < budget:
            if time.time() - start_time > DEADLINE:
                break

            # Select base vector: random
            base_idx = rng.integers(pop_size)
            # Select two distinct other vectors
            other_idxs = rng.choice(pop_size, size=2, replace=False)
            while other_idxs[0] == base_idx or other_idxs[1] == base_idx:
                other_idxs = rng.choice(pop_size, size=2, replace=False)

            # Differential evolution mutation
            mutant = X_current[base_idx] + F_mut * (X_current[other_idxs[0]] - X_current[other_idxs[1]])
            mutant = np.clip(mutant, lo, hi)

            # Crossover: binomial
            trial = X_current[base_idx].copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Replace if trial dominates base or base is worse
            if np.all(f_trial <= F_current[base_idx]) and np.any(f_trial < F_current[base_idx]):
                X_current[base_idx] = trial
                F_current[base_idx] = f_trial
            elif np.all(F_current[base_idx] <= f_trial) and np.any(F_current[base_idx] < f_trial):
                # base dominates trial, keep base
                pass
            else:
                # Non-dominated: replace if trial has better sum or random
                if np.sum(f_trial) < np.sum(F_current[base_idx]) or rng.random() < 0.5:
                    X_current[base_idx] = trial
                    F_current[base_idx] = f_trial

            # Update archive
            update_archive(trial, f_trial)

            # Occasionally inject a random solution to maintain diversity
            if used % 10 == 0 and used < budget:
                if time.time() - start_time > DEADLINE:
                    break
                rand_x = lo + span * rng.random(dim)
                rand_f = np.array(problem(rand_x))
                used += 1
                # Replace the worst solution in the population
                worst_idx = np.argmax(np.sum(F_current, axis=1))
                X_current[worst_idx] = rand_x
                F_current[worst_idx] = rand_f
                update_archive(rand_x, rand_f)

        # Final: ensure we haven't exceeded budget
        return X_current
# EVOLVE-BLOCK-END
