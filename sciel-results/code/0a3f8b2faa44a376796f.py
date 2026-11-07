"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: DE for global exploration, then multi-start
    Nelder-Mead for local exploitation. Exploits the smooth basin
    structure of Büche-Rastrigin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_calls, deadline, rng):
        """Nelder-Mead local search with call tracking."""
        dim = self.dim
        calls = 0
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        calls += 1

        # Add perturbation points
        for i in range(dim):
            pt = x0.copy()
            step = 0.5 if abs(x0[i]) < 1e-10 else 0.05 * abs(x0[i])
            pt[i] += step
            pt = np.clip(pt, -5, 5)
            simplex.append(pt)
            fvals.append(func(pt))
            calls += 1

        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        while calls < max_calls and time.time() < deadline:
            if len(simplex) != dim + 1:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expand
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contract
                beta = 0.5
                if fr < fvals[-1]:
                    xc = centroid + beta * (xr - centroid)
                else:
                    xc = centroid - beta * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= max_calls:
                            break

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

        return simplex[0], fvals[0], calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 275  # 275s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0

        # Phase 1: DE for global exploration (60% of budget)
        de_budget = int(budget * 0.6)
        pop_size = min(20, de_budget)
        if pop_size < 2:
            pop_size = 2

        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])
        calls_used += pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        F = 0.8
        CR = 0.9

        while calls_used < de_budget and time.time() < deadline:
            if time.time() > deadline - 5:
                break

            i = rng.integers(0, pop_size)
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == i:
                a = rng.integers(0, pop_size)
            while b == i or b == a:
                b = rng.integers(0, pop_size)
            while c == i or c == a or c == b:
                c = rng.integers(0, pop_size)

            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, -5, 5)

            j_rand = rng.integers(0, dim)
            trial = np.copy(pop[i])
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]

            trial_f = func(trial)
            calls_used += 1

            if trial_f < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()

        # Phase 2: Multi-start Nelder-Mead from top-k candidates
        remaining = budget - calls_used
        if remaining > 0 and time.time() < deadline - 2:
            # Get top-k diverse candidates from DE population
            k = min(5, pop_size)
            order = np.argsort(fitness)
            candidates = [pop[order[i]].copy() for i in range(k)]
            cand_fits = [fitness[order[i]] for i in range(k)]

            # Allocate remaining budget among k starts
            calls_per_start = max(1, remaining // k)

            for start_idx in range(k):
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                alloc = min(calls_per_start, budget - calls_used)
                if alloc < dim + 2:
                    alloc = dim + 2
                    if calls_used + alloc > budget:
                        break

                x_local, f_local, used = self._nelder_mead(
                    func, candidates[start_idx], alloc, deadline - 2, rng
                )
                calls_used += used

                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
