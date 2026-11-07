"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 combining front-biased initialization with multi-restart DE."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track all evaluated points for hypervolume
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            all_F.append(fc)
            used += 1
            return fc

        # Multi-restart DE with front-biased initialization
        n_restarts = 5
        pop_size = min(10, budget // n_restarts // 2)
        per_restart_budget = budget // n_restarts

        for r in range(n_restarts):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize population for this restart
            if r == 0:
                # Front-biased: spread x1 across its range, randomize others
                X = np.zeros((pop_size, dim))
                x1_vals = np.linspace(lo[0], hi[0], pop_size)
                X[:, 0] = x1_vals
                X[:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random((pop_size, dim - 1))
            else:
                # Initialize from best points of previous archives + random
                if len(all_F) > 0:
                    F_arr = np.array(all_F)
                    # Pick some good points (low sum) as seeds
                    sums = np.sum(F_arr, axis=1)
                    n_seeds = min(pop_size // 2, len(all_F))
                    if n_seeds > 0:
                        seed_idx = rng.choice(len(all_F), size=n_seeds, replace=False)
                        # We don't have all_X, so we need to re-evaluate or use a different strategy
                        # Let's just use random for now since we don't track X
                        X = lo + (hi - lo) * rng.random((pop_size, dim))
                    else:
                        X = lo + (hi - lo) * rng.random((pop_size, dim))
                else:
                    X = lo + (hi - lo) * rng.random((pop_size, dim))

            F = np.array([evaluate(x) for x in X])

            # DE loop within this restart
            max_iter = per_restart_budget - pop_size
            while used < budget and max_iter > 0:
                if time.time() > deadline:
                    break
                max_iter -= 1

                for i in range(pop_size):
                    if used >= budget or time.time() > deadline:
                        break

                    # DE/rand/1 mutation
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    while a == i or b == i or c == i:
                        a, b, c = rng.choice(pop_size, size=3, replace=False)

                    F_factor = 0.5 + 0.5 * rng.random()
                    v = X[a] + F_factor * (X[b] - X[c])
                    v = np.clip(v, lo, hi)

                    # Crossover
                    CR = 0.5
                    j_rand = rng.integers(dim)
                    trial = X[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    ft = evaluate(trial)

                    # Selection
                    if np.all(ft <= F[i]) and np.any(ft < F[i]):
                        X[i] = trial
                        F[i] = ft
                    elif np.all(F[i] <= ft) and np.any(F[i] < ft):
                        pass  # keep parent
                    else:
                        # Non-dominated: replace if trial has lower sum
                        if np.sum(ft) < np.sum(F[i]):
                            X[i] = trial
                            F[i] = ft

        return all_F
# EVOLVE-BLOCK-END
