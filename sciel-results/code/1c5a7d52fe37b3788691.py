"""Rewritten optimizer: population-based differential evolution (DE/rand/1) with adaptive mutation + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based DE/rand/1 with adaptive F, C_r, and final Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            calls[0] += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, budget_remaining):
            """Nelder-Mead local refinement from x0."""
            nonlocal calls, best_f, best_x
            if calls[0] >= budget_remaining or time.time() >= deadline:
                return

            x = np.array(x0, dtype=float)
            f0 = counted(x)

            simplex = np.empty((dim + 1, dim))
            simplex[0] = x
            fvals = np.empty(dim + 1)
            fvals[0] = f0

            for i in range(dim):
                p = x.copy()
                p[i] += 0.05
                p = np.clip(p, lo, hi)
                simplex[i + 1] = p
                fvals[i + 1] = counted(p)

            alpha, gamma, rho = 1.0, 2.0, 0.5

            while calls[0] < budget_remaining and time.time() < deadline:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                x_best = simplex[0]
                x_worst = simplex[-1]
                centroid = simplex[:-1].mean(axis=0)

                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = counted(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = counted(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = counted(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            fvals[i] = counted(simplex[i])

                if np.std(fvals) < 1e-14:
                    break

        # DE/rand/1 parameters
        pop_size = 20
        budget_de = int(self.budget * 0.85)
        budget_nm = self.budget - budget_de

        # Initialize population: mix of origin-centered and random
        pop = np.empty((pop_size, dim))
        pop[0] = np.zeros(dim)
        pop[1] = rng.uniform(-2, 2, dim)
        pop[2] = rng.uniform(2, 5, dim)
        pop[3] = rng.uniform(-5, -2, dim)
        for i in range(4, pop_size):
            pop[i] = rng.uniform(lo, hi, dim)

        fpop = np.empty(pop_size)
        for i in range(pop_size):
            fpop[i] = counted(pop[i])

        # Adaptive DE parameters
        F = 0.8
        C_r = 0.5
        no_improve = 0
        max_no_improve = 30

        best_idx = np.argmin(fpop)
        best_val = fpop[best_idx]

        while calls[0] < budget_de and time.time() < deadline:
            # Adaptive F: decrease when stagnant, increase when exploring
            if no_improve > max_no_improve:
                F = max(0.3, F - 0.05)
                C_r = min(0.9, C_r + 0.05)
            elif no_improve > max_no_improve // 2:
                F = max(0.4, F - 0.02)

            # Check for premature convergence
            if np.std(fpop) < 1e-10:
                # Re-seed worst half with random perturbations around best
                for i in range(pop_size // 2):
                    idx = np.argmax(fpop[i:]) + i if i < pop_size else i
                    pop[i] = pop[best_idx] + rng.standard_normal(dim) * 0.5
                    pop[i] = np.clip(pop[i], lo, hi)
                    fpop[i] = counted(pop[i])
                no_improve = 0
                continue

            # Generate offspring
            for i in range(pop_size):
                if calls[0] >= budget_de or time.time() >= deadline:
                    break

                # Select three distinct random individuals
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)

                # Binomial crossover
                r = rng.random()
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() <= C_r or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                trial = np.clip(trial, lo, hi)
                f_trial = counted(trial)

                # Selection
                if f_trial <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = f_trial

            # Track improvement
            new_best_idx = np.argmin(fpop)
            new_best_val = fpop[new_best_idx]

            if new_best_val < best_val - 1e-12:
                best_val = new_best_val
                best_idx = new_best_idx
                no_improve = 0
                # Slight increase in F on improvement
                F = min(0.9, F + 0.02)
            else:
                no_improve += 1

        # Phase 2: Nelder-Mead polish from the best point
        if calls[0] < self.budget and time.time() < deadline:
            refine_budget = self.budget - calls[0]
            if best_x is not None:
                nelder_mead(best_x, refine_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
