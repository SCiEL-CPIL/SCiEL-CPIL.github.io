"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-style evolutionary strategy with
    deadline guard and tracked archive for hypervolume computation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initialize population
        pop_size = min(20, self.budget // 4)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        def dominates(a, b):
            """Return True if a dominates b (all <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_arr):
            """Return list of fronts, each a list of indices."""
            n = len(F_arr)
            fronts = []
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # nd[i] = number of solutions dominating i
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif dominates(F_arr[j], F_arr[i]):
                        nd[i] += 1
            front = [i for i in range(n) if nd[i] == 0]
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort to find Pareto fronts
            fronts = non_dominated_sort(F)

            # Select parents from the first front (Pareto optimal)
            if len(fronts) > 0:
                parent_pool = fronts[0]
            else:
                parent_pool = list(range(len(F)))

            parent_idx = parent_pool[rng.integers(len(parent_pool))]
            parent = X[parent_idx]

            # Generate child with adaptive step size
            step_size = 0.1
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Record point
            all_X.append(child)
            all_F.append(fc)

            # Replace worst solution in population using dominance-based replacement
            worst_idx = None
            for i in range(len(F)):
                if dominates(fc, F[i]):
                    worst_idx = i
                    break
            if worst_idx is None:
                # Use crowding distance approximation: prefer solutions in less crowded regions
                # Simple heuristic: pick the solution with the largest sum of objectives
                worst_idx = int(np.argmax(np.sum(F, axis=1)))

            X[worst_idx] = child
            F[worst_idx] = fc

        return all_X
# EVOLVE-BLOCK-END
