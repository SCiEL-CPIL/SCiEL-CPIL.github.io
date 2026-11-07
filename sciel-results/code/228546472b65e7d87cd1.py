"""Multi-objective optimizer on RE34 using differential evolution with error repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with error repair for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, keep safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(20, budget // 4)
        if pop_size < 2:
            pop_size = max(2, budget // 4)

        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE parameters
        F_mut = 0.5
        CR = 0.5

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Select three distinct parents different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_a + F * (x_b - x_c)
                v = X[a] + F_mut * (X[b] - X[c])

                # Error Repair: reflect out-of-bounds values
                for d in range(dim):
                    if v[d] < lo[d]:
                        v[d] = 2 * lo[d] - v[d]
                    elif v[d] > hi[d]:
                        v[d] = 2 * hi[d] - v[d]
                    # Final clamp for safety
                    v[d] = np.clip(v[d], lo[d], hi[d])

                # Crossover: Binomial crossover
                r = rng.random(dim)
                j_rand = rng.integers(dim)
                u = np.where(r <= CR, v, X[i])
                u[j_rand] = v[j_rand]  # Ensure at least one gene from v

                # Evaluate
                f_u = problem(u)
                used += 1

                # Selection: keep if u dominates X[i], or if non-dominated and better sum
                if np.all(f_u <= F[i]) and np.any(f_u < F[i]):
                    # u dominates X[i]
                    X[i] = u
                    F[i] = f_u
                elif not (np.all(F[i] <= f_u) and np.any(F[i] < f_u)):
                    # Neither dominates; use sum heuristic
                    if np.sum(f_u) < np.sum(F[i]):
                        X[i] = u
                        F[i] = f_u

        return X
# EVOLVE-BLOCK-END
