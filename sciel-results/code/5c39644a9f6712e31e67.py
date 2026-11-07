"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: larger for better hypervolume coverage
        pop_size = min(40, self.budget // 4)
        X = np.empty((pop_size, dim))
        # 1/3 random, 1/3 mid-range, 1/3 boundary/corner points
        n_rand = pop_size // 3
        n_mid = pop_size // 3
        n_bound = pop_size - n_rand - n_mid
        X[:n_rand] = lo + span * rng.random((n_rand, dim))
        X[n_rand:n_rand+n_mid] = lo + span * (0.3 + 0.4 * rng.random((n_mid, dim)))
        for i in range(n_rand + n_mid, pop_size):
            # Mix of corners and edge points
            r = rng.random()
            if r < 0.5:
                corners = rng.random(dim) < 0.5
                X[i] = np.where(corners, lo, hi)
            else:
                # Edge point: most at random, one or two at bounds
                X[i] = lo + span * rng.random(dim)
                n_edges = rng.integers(1, 3)
                edge_idx = rng.choice(dim, size=n_edges, replace=False)
                for ei in edge_idx:
                    X[i, ei] = lo[ei] if rng.random() < 0.5 else hi[ei]
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points (vectorized for speed)
            n = len(F)
            # F[i] dominates F[j] if all(F[i] <= F[j]) and any(F[i] < F[j])
            # Use broadcasting: F[i, None, :] <= F[None, j, :] 
            leq = np.all(F[:, None, :] <= F[None, :, :], axis=2)  # n x n
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)   # n x n
            dominates = leq & lt  # dominates[i,j] = True if F[i] dominates F[j]
            np.fill_diagonal(dominates, False)
            dominated = np.any(dominates, axis=0)  # dominated[j] = True if any i dominates j

            non_dom_idx = np.where(~dominated)[0]
            
            # Select parent from non-dominated set
            if len(non_dom_idx) > 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                parent_idx = rng.integers(n)

            # Adaptive mutation: step size decreases over time
            progress = used / self.budget
            base_step = 0.12 * (1.0 - 0.8 * progress)
            
            r = rng.random()
            if r < 0.25 and len(F) >= 2 and len(non_dom_idx) > 1:
                # Crossover with another non-dominated point
                other_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                alpha = rng.random()
                child = alpha * X[parent_idx] + (1 - alpha) * X[other_idx]
                child += 0.03 * span * rng.standard_normal(dim)
            elif r < 0.65:
                # Gaussian mutation
                child = X[parent_idx] + base_step * span * rng.standard_normal(dim)
            else:
                # Larger jump to explore
                child = X[parent_idx] + 0.25 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point: prefer dominated points, then worst hypervolume contribution
            # Use sum of normalized objectives as proxy
            worst_score = np.sum(F, axis=1)
            for i in range(n):
                if dominated[i]:
                    worst_score[i] += 1e10
            worst_idx = int(np.argmax(worst_score))
            X[worst_idx] = child
            F[worst_idx] = fc

        # Final polish: local refinement on non-dominated points
        if time.time() < deadline - 5.0:
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            non_dom_idx = np.where(~dominated)[0]
            
            for idx in non_dom_idx:
                if used >= self.budget or time.time() > deadline:
                    break
                # Try small perturbations around this point
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    child = X[idx] + 0.02 * span * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    # Replace if it dominates or is non-dominated and better sum
                    if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                        X[idx] = child
                        F[idx] = fc
                    elif not np.any(np.all(F <= fc) and np.any(F < fc)):
                        # Check if fc is non-dominated
                        is_dom = False
                        for k in range(n):
                            if k != idx and np.all(F[k] <= fc) and np.any(F[k] < fc):
                                is_dom = True
                                break
                        if not is_dom and np.sum(fc) < np.sum(F[idx]):
                            X[idx] = child
                            F[idx] = fc

        return X
# EVOLVE-BLOCK-END
