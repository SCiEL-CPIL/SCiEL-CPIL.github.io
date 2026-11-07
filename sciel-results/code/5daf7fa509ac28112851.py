"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and improved selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: mix of random and boundary points
        pop_size = 16
        X = np.empty((pop_size, dim))
        # Half random interior
        X[:pop_size // 2] = lo + span * rng.random((pop_size // 2, dim))
        # Half at random corners (mix of lo and hi per dimension)
        for i in range(pop_size // 2, pop_size):
            corners = (rng.random(dim) < 0.5)
            X[i] = np.where(corners, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points
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
            
            # Select parent from non-dominated set with tournament-like preference
            if len(non_dom_idx) > 0:
                # Prefer points that are well-spread (use sum of normalized objectives)
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = np.where(f_max - f_min > 0, f_max - f_min, 1.0)
                F_norm = (F - f_min) / f_range
                scores = F_norm[:, non_dom_idx].sum(axis=0)
                # Use softmax-like selection
                scores = scores - scores.max()
                probs = np.exp(-1.0 * scores)
                probs /= probs.sum()
                parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=probs)]
            else:
                parent_idx = rng.integers(n)

            # Adaptive mutation: mix of Gaussian and crossover
            r = rng.random()
            if r < 0.3 and len(F) >= 2:
                # Crossover with another non-dominated point
                other_idx = non_dom_idx[rng.integers(len(non_dom_idx))] if len(non_dom_idx) > 1 else rng.integers(n)
                alpha = rng.random()
                child = alpha * X[parent_idx] + (1 - alpha) * X[other_idx]
                # Add small mutation
                child += 0.05 * span * rng.standard_normal(dim)
            elif r < 0.7:
                # Gaussian mutation with adaptive step
                step_size = 0.1 * (1.0 - 0.5 * used / self.budget)
                child = X[parent_idx] + step_size * span * rng.standard_normal(dim)
            else:
                # LARGER jump to explore
                child = X[parent_idx] + 0.3 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point in current population
            # Use a composite score: prefer replacing dominated points, then worst hypervolume contribution
            worst_score = np.sum(F, axis=1)
            # Penalize dominated points heavily
            for i in range(n):
                if dominated[i]:
                    worst_score[i] += 1e10
            worst_idx = int(np.argmax(worst_score))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Periodically re-seed with new random points to maintain diversity
            if used % 20 == 0 and used < self.budget - 10:
                # Replace one random point with a fresh random one
                repl_idx = rng.integers(pop_size)
                X[repl_idx] = lo + span * rng.random(dim)
                F[repl_idx] = problem(X[repl_idx])
                used += 1

        return X
# EVOLVE-BLOCK-END
