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
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use a local RNG for speed and reproducibility
        rng = np.random.default_rng(42)

        # Population size: Small enough for fast dominance checks, large enough for coverage
        # 40 is a good balance: 40^2 = 1600 ops for dominance, allowing many generations
        pop_size = 40

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Vectorized dominance check
        # Returns a boolean mask 'dominated' where True means the point is dominated by at least one other point
        def get_dominated_mask(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            
            # F_j is (N, 1, M), F_i is (1, N, M)
            F_j = F[:, np.newaxis, :]
            F_i = F[np.newaxis, :, :]
            
            # i dominates j if all(F[i] <= F[j]) and any(F[i] < F[j])
            # Here we check if j is dominated by i
            # Let's compute matrix D where D[i,j] is True if i dominates j
            le = F_i <= F_j  # (1, N, M) <= (N, 1, M) -> (N, N, M) ? No, broadcasting:
            # F_i: (1, N, M)
            # F_j: (N, 1, M)
            # Result: (N, N, M) where element [i, j, k] is F_i[i, k] <= F_j[j, k]
            # Wait, F_i is indexed by i (row), F_j by j (col)?
            # Let's be precise.
            # F is (N, M).
            # We want D[i, j] = True if i dominates j.
            # Condition: for all k, F[i, k] <= F[j, k] AND for at least one k, F[i, k] < F[j, k].
            
            # F[i] is row i. F[j] is row j.
            # F_rows = F[:, np.newaxis, :] # (N, 1, M) -> represents F[i] for all i
            # F_cols = F[np.newaxis, :, :] # (1, N, M) -> represents F[j] for all j
            
            # le[i, j, k] = F[i, k] <= F[j, k]
            le = F[:, np.newaxis, :] <= F[np.newaxis, :, :]  # (N, N, M)
            lt = F[:, np.newaxis, :] < F[np.newaxis, :, :]   # (N, N, M)
            
            all_le = np.all(le, axis=2) # (N, N)
            any_lt = np.any(lt, axis=2) # (N, N)
            
            dom_matrix = all_le & any_lt # dom_matrix[i, j] is True if i dominates j
            
            # A point j is dominated if there exists an i such that dom_matrix[i, j] is True
            dominated = np.any(dom_matrix, axis=0)
            
            return dominated

        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Identify dominated and non-dominated points
            dominated_mask = get_dominated_mask(F)
            nd_indices = np.where(~dominated_mask)[0]

            if len(nd_indices) == 0:
                # Should not happen, but fallback to all
                nd_indices = np.arange(pop_size)

            # Selection: Pick parents from non-dominated set
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
            # We want to maintain a diverse set of non-dominated points.
            # 1. If child is non-dominated (w.r.t current F), it's a candidate for addition.
            # 2. If child is dominated, it's only added if it improves the worst point or diversity.
            
            # Check if child is dominated by any point in current F
            # Re-use vectorized logic for single point
            le_c = F <= fc[np.newaxis, :]  # (N, M)
            lt_c = F < fc[np.newaxis, :]   # (N, M)
            doms_child = np.all(le_c, axis=1) & np.any(lc_c, axis=1)
            child_is_dominated = np.any(doms_child)

            # Strategy:
            # If child is non-dominated, replace the worst point (highest sum) OR a dominated point.
            # If child is dominated, replace the worst dominated point if child is better than it in some way?
            # Simpler robust strategy:
            # Always replace the point with the highest "score".
            # Score can be:
            # - If non-dominated: 0 (best)
            # - If dominated: sum of objectives (higher is worse)
            # But we want to keep diversity.
            
            # Let's stick to a simpler heuristic that worked in Inspiration 1:
            # Compute sums
            F_sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(F_sums))
            worst_sum = F_sums[worst_idx]
            child_sum = np.sum(fc)

            if not child_is_dominated:
                # Child is non-dominated. It is valuable.
                # Replace the worst point (highest sum) to ensure we keep the best non-dominated front.
                # Actually, if child is non-dominated, it might be better to replace a dominated point if available,
                # or the worst non-dominated point if we are full.
                # Since we have a fixed pop_size, we must replace someone.
                # Replacing the point with the highest sum is a safe heuristic for minimization.
                X[worst_idx] = child
                F[worst_idx] = fc
            else:
                # Child is dominated.
                # Only replace if it is better than the worst point in the population.
                if child_sum < worst_sum:
                    X[worst_idx] = child
                    F[worst_idx] = fc
                else:
                    # Try to replace a dominated point if the child is "less dominated" or just random exploration?
                    # To maintain diversity, if we have dominated points, we might replace a random one
                    # if the child is not too bad. But usually, we discard dominated children unless they improve the worst.
                    # However, to avoid stagnation, let's replace a random dominated point if available,
                    # with a small probability, to allow exploration.
                    if np.any(dominated_mask) and rng.random() < 0.1:
                        dom_points = np.where(dominated_mask)[0]
                        replace_idx = dom_points[rng.integers(len(dom_points))]
                        # Only replace if child is better than the replaced point in sum?
                        if child_sum < F_sums[replace_idx]:
                            X[replace_idx] = child
                            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
