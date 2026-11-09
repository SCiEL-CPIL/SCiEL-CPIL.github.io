"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer for DTLZ4 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, with safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Use the global numpy random state (seeded by evaluator)
        # We'll use np.random directly since evaluator calls np.random.seed(seed)

        # Initial population
        pop_size = min(20, max(10, self.budget // 10))
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (a is better in at least one obj and no worse in all)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(F, i):
            """Check if F[i] is dominated by any other point in F."""
            fi = F[i]
            for j in range(len(F)):
                if j == i:
                    continue
                if dominates(F[j], fi):
                    return True
            return False

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5.0:  # 5s safety margin
                break

            # Find non-dominated set
            non_dom_mask = np.array([not is_dominated(F, i) for i in range(len(F))])
            non_dom_idx = np.where(non_dom_mask)[0]

            # If all dominated (shouldn't happen), use all
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parents from non-dominated set
            n_parents = min(2, len(non_dom_idx))
            if n_parents < 2:
                # If only one non-dominated, pick it and a random one
                p1_idx = non_dom_idx[0]
                p2_idx = np.random.randint(0, len(F))
            else:
                # Tournament selection among non-dominated
                candidates = non_dom_idx[:min(10, len(non_dom_idx))]
                p1_idx = candidates[np.random.randint(0, len(candidates))]
                p2_idx = candidates[np.random.randint(0, len(candidates))]
                if p2_idx == p1_idx:
                    p2_idx = candidates[(np.random.randint(0, len(candidates)) + 1) % len(candidates)]

            # Crossover: SBX-like (blending)
            alpha = np.random.random(self.dim)
            child1 = (1 - alpha) * X[p1_idx] + alpha * X[p2_idx]
            child2 = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]

            # Mutation: polynomial-like
            def mutate(x):
                # Gaussian mutation with adaptive step
                step = 0.1 * span
                return np.clip(x + step * np.random.standard_normal(self.dim), lo, hi)

            child1 = mutate(child1)
            child2 = mutate(child2)

            # Evaluate children
            fc1 = problem(child1)
            used += 1
            if time.time() > deadline - 5.0:
                # Replace worst
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child1, fc1
                break
            fc2 = problem(child2)
            used += 1

            # Replace worst individuals
            for child, fc in [(child1, fc1), (child2, fc2)]:
                if used >= self.budget:
                    break
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child, fc

        # Final: return the population (return value is ignored by evaluator)
        return X

# EVOLVE-BLOCK-END
