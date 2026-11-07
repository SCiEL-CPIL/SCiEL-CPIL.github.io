"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """Multi-start DE + Nelder-Mead with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Differential Evolution for global exploration ---
        # Population size
        pop_size = max(8, 4 * dim)
        # Initialize population uniformly in the box
        pop = rng.uniform(-5.0, 5.0, (pop_size, dim))
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            if not time_ok() or i >= budget - 2:
                break
        if len(fitness) < pop_size:
            pop = pop[:len(fitness)]
            pop_size = len(fitness)
            fitness = fitness[:pop_size]

        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8   # scale factor
        CR = 0.9  # crossover rate

        calls_used = len(fitness)
        # Run DE until budget or time limit
        while calls_used < budget - 1 and time_ok():
            # Differential mutation
            i = rng.integers(0, pop_size)
            j1 = rng.integers(0, pop_size)
            j2 = rng.integers(0, pop_size)
            while j1 == i or j2 == i or j1 == j2:
                j1 = rng.integers(0, pop_size)
                j2 = rng.integers(0, pop_size)

            v = pop[j1] + F * (pop[j2] - pop[i])
            v = np.clip(v, -5.0, 5.0)

            # Binomial crossover
            trial = pop[i].copy()
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = np.clip(trial, -5.0, 5.0)

            fy = func(trial)
            calls_used += 1

            if fy <= fitness[i]:
                pop[i] = trial
                fitness[i] = fy
                if fy < best_f:
                    best_f = fy
                    best_x = trial.copy()
                    best_idx = i

        # --- Phase 2: Local refinement with Nelder-Mead on the best point ---
        # Use remaining budget for local search
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            # Nelder-Mead simplex
            simplex = [best_x.copy()]
            for k in range(dim):
                dx = best_x.copy()
                dx[k] += 0.1
                dx = np.clip(dx, -5.0, 5.0)
                simplex.append(dx)
            simplex = np.array(simplex)
            simplex_f = np.empty(dim + 1)
            simplex_f[0] = best_f
            for k in range(1, dim + 1):
                simplex_f[k] = func(simplex[k])
                calls_used += 1
                if not time_ok():
                    break

            if calls_used < budget and time_ok():
                # Nelder-Mead iterations
                max_iter = min(remaining - 1, 200)
                for _ in range(max_iter):
                    if not time_ok() or calls_used >= budget:
                        break
                    # Sort by fitness
                    order = np.argsort(simplex_f)
                    simplex = simplex[order]
                    simplex_f = simplex_f[order]

                    # Reflection
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid - 1.0 * (simplex[-1] - centroid)
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = func(xr)
                    calls_used += 1

                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid - 2.0 * (simplex[-1] - centroid)
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = func(xe)
                        calls_used += 1
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[-1]:
                            # Outside contraction
                            xc = centroid + 0.5 * (simplex[-1] - centroid)
                            xc = np.clip(xc, -5.0, 5.0)
                            fc = func(xc)
                            calls_used += 1
                            if fc <= fr:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for k in range(1, dim + 1):
                                    simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                    simplex_f[k] = func(simplex[k])
                                    calls_used += 1
                                    if not time_ok():
                                        break
                        else:
                            # Inside contraction
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                            xc = np.clip(xc, -5.0, 5.0)
                            fc = func(xc)
                            calls_used += 1
                            if fc < simplex_f[-1]:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for k in range(1, dim + 1):
                                    simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                    simplex_f[k] = func(simplex[k])
                                    calls_used += 1
                                    if not time_ok():
                                        break

                    # Update best
                    min_idx = int(np.argmin(simplex_f))
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

        return best_x, best_f
