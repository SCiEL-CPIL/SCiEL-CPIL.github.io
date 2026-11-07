"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS initialization and Hypervolume-Indicative Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s wall-clock limit, return early if exceeded

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Latin Hypercube Sampling for initialization
        pop_size = min(20, budget // 4)
        if pop_size < 4:
            pop_size = 4

        # LHS initialization
        X = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.arange(pop_size)
            np.random.shuffle(perm)
            X[np.arange(pop_size), j] = (perm + np.random.random(pop_size)) / pop_size
        X = lo + (hi - lo) * X

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Differential Evolution style mutation with hypervolume-based selection
        F_CR = 0.5
        F_F = 0.8
        rng = np.random.default_rng(0)

        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Find non-dominated set
            nondom_mask = self._non_dominated(F)
            non_dom_idx = np.where(nondom_mask)[0]

            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parents from non-dominated set
            n_parents = min(len(non_dom_idx), 3)
            parent_idx = rng.choice(non_dom_idx, size=n_parents, replace=False)
            r1, r2, r3 = parent_idx[0], parent_idx[1], parent_idx[2]

            # DE/3 mutation
            v = F[r1] + F_CR * (F[r2] - F[r3])

            # Binary crossover
            cross_mask = rng.random(dim) < 0.5
            j_rand = rng.integers(dim)
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, v, X[r1])
            trial = np.clip(trial, lo, hi)

            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Replacement: keep if it improves hypervolume or is non-dominated
            # Simple approach: replace worst point by sum of objectives
            worst_idx = np.argmax(np.sum(F, axis=1))

            # Check if trial is better than worst
            if np.all(f_trial <= F[worst_idx]) or not np.any(np.all(F[worst_idx] <= f_trial) & np.any(F[worst_idx] < f_trial)):
                X[worst_idx] = trial
                F[worst_idx] = f_trial

            # Also try a random perturbation around best points sometimes
            if used % 5 == 0 and used < budget - 1:
                # Pick a random point and perturb
                base_idx = rng.integers(len(F))
                child = X[base_idx] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                f_child = problem(child)
                used += 1

                # Replace worst
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.all(f_child <= F[worst_idx]) or not np.any(np.all(F[worst_idx] <= f_child) & np.any(F[worst_idx] < f_child)):
                    X[worst_idx] = child
                    F[worst_idx] = f_child

        return X

    def _non_dominated(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated
# EVOLVE-BLOCK-END
