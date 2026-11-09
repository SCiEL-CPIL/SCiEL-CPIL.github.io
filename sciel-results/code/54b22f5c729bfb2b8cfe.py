"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4: simple non-dominated parent mutation with
       a small elite archive and occasional restarts to escape stagnation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Small population to keep per-evaluation overhead low
        pop_size = 10

        # DTLZ4 structure: last (n_obj - 1) variables are "position" variables
        # that should be pushed to 0 or 1 for Pareto-optimal solutions.
        # First (dim - n_obj + 1) variables are "shape" variables.
        n_pos = self.n_obj - 1
        pos_start = self.dim - n_pos

        def non_dominated_mask(F):
            """Return boolean array: True for non-dominated points."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        def dominated_mask(F):
            """Return boolean array: True for dominated points."""
            n = len(F)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom[i] = True
                        break
            return dom

        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Elite archive: keep all non-dominated points seen so far
        # (capped to avoid unbounded growth)
        elite_X = []
        elite_F = []

        def update_elite(x, f):
            """Add point to elite if not dominated by any elite point."""
            nonlocal elite_X, elite_F
            if len(elite_F) == 0:
                elite_X.append(x)
                elite_F.append(f)
                return
            ef = np.array(elite_F)
            # Check if x is dominated by any elite
            if np.any(np.all(ef <= f, axis=1) & np.any(ef < f, axis=1)):
                return
            # Remove elites dominated by x
            keep = []
            for i in range(len(elite_F)):
                if not (np.all(f <= elite_F[i]) and np.any(f < elite_F[i])):
                    keep.append(i)
            elite_X = [elite_X[i] for i in keep]
            elite_F = [elite_F[i] for i in keep]
            elite_X.append(x)
            elite_F.append(f)
            # Cap elite size
            if len(elite_X) > 50:
                # Keep the 50 most spread out points (by sum of objectives)
                sums = np.array([np.sum(f) for f in elite_F])
                # Actually, just keep first 50 non-dominated
                elite_X = elite_X[:50]
                elite_F = elite_F[:50]

        # Update elite with initial population
        for i in range(pop_size):
            update_elite(X[i], F[i])

        def generate_child(parent):
            """Generate a child from parent with DTLZ4-aware mutation."""
            child = parent.copy()
            # Shape variables: small Gaussian perturbation
            shape_noise = 0.05 * range_[:pos_start] * rng.standard_normal(pos_start)
            child[:pos_start] = np.clip(child[:pos_start] + shape_noise, lo[:pos_start], hi[:pos_start])
            # Position variables: push toward 0 or 1
            for i in range(pos_start, self.dim):
                if rng.random() < 0.7:
                    child[i] = 0.0 if rng.random() < 0.5 else 1.0
                else:
                    child[i] = np.clip(child[i] + 0.1 * range_[i] * rng.standard_normal(), lo[i], hi[i])
            return child

        no_improve_count = 0
        best_sum = np.sum(np.min(F, axis=1))

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_mask = non_dominated_mask(F)
            if np.any(nd_mask):
                parents = X[nd_mask]
            else:
                parents = X

            # Select a parent randomly from non-dominated set
            parent = parents[rng.integers(len(parents))]
            child = generate_child(parent)
            fc = np.array(problem(child))
            used += 1

            # Check if child improves the archive
            child_sum = np.sum(fc)
            if child_sum < best_sum:
                best_sum = child_sum
                no_improve_count = 0
            else:
                no_improve_count += 1

            update_elite(child, fc)

            # Replace the worst point in the population
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Occasional restart if stuck: reseed some points from elite
            if no_improve_count > 30 and len(elite_X) > 0:
                # Replace 3 random population members with elite points + mutation
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    e_idx = rng.integers(len(elite_X))
                    e_child = generate_child(elite_X[e_idx])
                    e_fc = np.array(problem(e_child))
                    used += 1
                    update_elite(e_child, e_fc)
                    w_idx = int(np.argmax(np.sum(F, axis=1)))
                    X[w_idx] = e_child
                    F[w_idx] = e_fc
                no_improve_count = 0

        return X
# EVOLVE-BLOCK-END
