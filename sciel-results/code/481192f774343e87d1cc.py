"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7 with crossover and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use a local RNG for speed and reproducibility
        rng = np.random.default_rng(42)

        # Population size: small enough for fast dominance checks, large enough for coverage
        pop_size = 30

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Fast non-dominated check using vectorized operations
            F_j = F[:, np.newaxis, :]  # (N, 1, M)
            F_i = F[np.newaxis, :, :]  # (1, N, M)

            le = F_j <= F_i
            lt = F_j < F_i

            all_le = np.all(le, axis=2)
            any_lt = np.any(lt, axis=2)

            dom = all_le & any_lt
            np.fill_diagonal(dom, False)

            dominated = np.any(dom, axis=0)
            nd_indices = np.where(~dominated)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Selection: pick parents from non-dominated set
            # With high probability, use crossover between two parents
            if len(nd_indices) >= 2 and rng.random() < 0.7:
                p1_idx = nd_indices[rng.integers(len(nd_indices))]
                p2_idx = nd_indices[rng.integers(len(nd_indices))]
                parent1 = X[p1_idx]
                parent2 = X[p2_idx]
                
                # Uniform crossover
                mask = rng.random(self.dim) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                # Mutation only
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
                parent = X[parent_idx]
                
                # Adaptive step size: start larger, shrink over time
                progress = used / self.budget
                step_size = 0.15 * (1 - 0.5 * progress)
                
                child = parent + step_size * span * rng.standard_normal(self.dim)

            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement strategy:
            # 1. Find the worst point in the current population (highest sum of objectives)
            # 2. Replace it with the child if the child is better in sum, or if child is non-dominated
            # 3. Otherwise, still replace to maintain diversity, but prefer replacing dominated points

            # Compute sums
            F_sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(F_sums))
            worst_sum = F_sums[worst_idx]
            child_sum = np.sum(fc)

            # Check if child is dominated by any point in F
            # F_j is (N, 1, M), fc is (M,)
            # We need to check if any F[j] dominates fc
            # Condition: all(F[j] <= fc) and any(F[j] < fc)
            le_c = F <= fc[np.newaxis, :]  # (N, M)
            lt_c = F < fc[np.newaxis, :]   # (N, M)
            doms_child = np.all(le_c, axis=1) & np.any(lt_c, axis=1)
            child_is_dominated = np.any(doms_child)

            if child_sum < worst_sum or not child_is_dominated:
                # Child is better than worst or non-dominated, so replace worst
                X[worst_idx] = child
                F[worst_idx] = fc
            else:
                # Child is dominated and worse than worst in sum.
                # Try to find a dominated point to replace.
                # dominated is already computed above for the current F.
                if np.any(dominated):
                    dom_points = np.where(dominated)[0]
                    # Replace a random dominated point
                    replace_idx = dom_points[rng.integers(len(dom_points))]
                    X[replace_idx] = child
                    F[replace_idx] = fc
                else:
                    # No dominated points, replace worst anyway to explore
                    X[worst_idx] = child
                    F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
