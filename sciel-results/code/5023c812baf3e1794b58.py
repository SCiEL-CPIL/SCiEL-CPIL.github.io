"""Multi-objective optimizer for RE34 using Multi-Start Differential Evolution with archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Multi-Start Differential Evolution with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Use the global numpy random state since the evaluator seeds it
        # rng = np.random

        # Structured initialization: corners, midpoints, random
        # For 5D with bounds [1,3], create diverse initial points
        init_points = []
        
        # All 32 corners
        for i in range(32):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)
        
        # Midpoints
        init_points.append((lo + hi) / 2)
        
        # Add random points to fill up to population size
        pop_size = 32
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * np.random.random(dim))
        
        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        def is_dominated(f, archive_F):
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Remove dominated solutions from archive
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            return new_archive_X, new_archive_F

        # Differential Evolution parameters
        F_factor = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parents from non-dominated solutions in archive
            if len(archive_X) > 0:
                parent_idx = np.random.randint(len(archive_X))
                parent = archive_X[parent_idx]
            else:
                parent = X[np.random.randint(len(X))]

            # Differential Evolution: select 3 random distinct indices
            # Ensure they are distinct from each other
            idx1, idx2, idx3 = np.random.choice(len(X), size=3, replace=False)
            
            # Create trial vector
            r = np.random.random(dim)
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            trial = np.where(r < CR, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Replace worst solution in population if trial is better
            # Use a simple dominance check or sum-based comparison
            # Find worst by sum of objectives (simple heuristic)
            worst_idx = np.argmax(np.sum(F, axis=1))
            
            # If trial dominates worst, replace
            if np.all(ft <= F[worst_idx]) and np.any(ft < F[worst_idx]):
                X[worst_idx] = trial
                F[worst_idx] = ft
            # If trial is non-dominated (not dominated by archive), replace worst non-archive solution
            elif not is_dominated(ft, archive_F):
                # Find the worst solution not in archive
                # Simple check: if trial is not in archive, and worst is in archive, find next worst
                # To keep it fast, just replace worst if worst is not a unique archive member?
                # Simpler: replace worst if it's not the best in archive?
                # Let's just replace the worst index if the trial is non-dominated.
                # This ensures the population stays diverse and contains good solutions.
                X[worst_idx] = trial
                F[worst_idx] = ft

        return X
# EVOLVE-BLOCK-END
