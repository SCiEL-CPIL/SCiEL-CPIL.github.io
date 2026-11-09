"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 with structural exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 275

        rng = np.random.default_rng(7)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        eval_count = [0]

        def evaluate(x):
            if eval_count[0] >= budget:
                return None
            eval_count[0] += 1
            return problem(x)

        # Phase 1: Structured initialization with minimal noise in x[1:]
        n_init = 40
        X_list = []
        F_list = []

        for i in range(n_init):
            x = np.zeros(dim)
            x[0] = i / max(n_init - 1, 1)
            x[1:] = rng.uniform(0, 0.003, dim - 1)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is None:
                break
            X_list.append(x)
            F_list.append(f)

        # Phase 2: Unified local search with exponential decay
        pop_size = 100
        max_iter = budget - eval_count[0]
        step_scale = 1.0

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if eval_count[0] >= budget:
                break

            step_scale *= 0.995
            idx = rng.integers(len(X_list))
            parent = X_list[idx].copy()
            child = parent.copy()

            # Aggressively drive x[1:] toward 0
            for j in range(1, dim):
                if child[j] > 0.001:
                    child[j] = child[j] * rng.uniform(0.3, 0.7) + rng.uniform(-0.002, 0.002) * step_scale
                    child[j] = max(0.0, min(1.0, child[j]))
                elif child[j] > 0:
                    child[j] = max(0.0, child[j] + rng.uniform(-0.001, 0.001) * step_scale)

            # Perturb x[0]
            child[0] = parent[0] + rng.uniform(-0.03, 0.03) * step_scale
            child[0] = max(0.0, min(1.0, child[0]))

            fc = evaluate(child)
            if fc is None:
                break

            # Diversity-aware replacement
            F_arr = np.array(F_list)
            sums = np.sum(F_arr, axis=1)
            worst_idx = int(np.argmax(sums))

            # Check if child is dominated by any point
            dominated = False
            for k in range(len(F_list)):
                if np.all(F_list[k] <= fc) and np.any(F_list[k] < fc):
                    dominated = True
                    break

            if not dominated and np.sum(fc) < sums[worst_idx]:
                X_list[worst_idx] = child
                F_list[worst_idx] = fc
            elif not dominated and len(F_list) < pop_size:
                X_list.append(child)
                F_list.append(fc)

        # Phase 3: Final polishing with tiny steps
        remaining = budget - eval_count[0]
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if eval_count[0] >= budget:
                break

            idx = rng.integers(len(X_list))
            parent = X_list[idx].copy()
            child = parent.copy()

            # Very small perturbations
            child[0] = parent[0] + rng.uniform(-0.005, 0.005)
            child[0] = max(0.0, min(1.0, child[0]))

            for j in range(1, dim):
                child[j] = max(0.0, child[j] * rng.uniform(0.8, 1.0) + rng.uniform(-0.0005, 0.0005))

            fc = evaluate(child)
            if fc is None:
                break

            F_arr = np.array(F_list)
            sums = np.sum(F_arr, axis=1)
            worst_idx = int(np.argmax(sums))

            if np.sum(fc) < sums[worst_idx]:
                X_list[worst_idx] = child
                F_list[worst_idx] = fc

        return X_list
# EVOLVE-BLOCK-END
