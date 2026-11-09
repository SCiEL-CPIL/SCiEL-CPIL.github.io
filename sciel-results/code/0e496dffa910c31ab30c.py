# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 300.0 - 5.0  # safety margin before 300s kill

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # ---- Phase 1: population-based DE global search ----
        pop_size = 24
        F = 0.8
        CR = 0.7

        # initial population spread over the box
        pop = rng.uniform(-5.0, 5.0, (pop_size, dim))
        fpop = np.empty(pop_size)
        for i in range(pop_size):
            fpop[i] = func(pop[i])

        best_idx = int(np.argmin(fpop))
        x_best = pop[best_idx].copy()
        f_best = fpop[best_idx]

        def still_going():
            return time.time() < self.deadline

        calls = pop_size
        t0 = time.time()
        while calls < budget and still_going():
            base = int(rng.integers(0, pop_size))
            a = int(rng.integers(0, pop_size))
            b = int(rng.integers(0, pop_size))
            c = int(rng.integers(0, pop_size))
            if len({a, b, c, base}) < 4:
                continue
            v = pop[base] + F * (pop[a] - pop[b] + pop[c] - pop[base])
            v = np.clip(v, -5.0, 5.0)
            trial = np.empty(dim)
            j0 = int(rng.integers(0, dim))
            for j in range(dim):
                if rng.random() < CR or j == j0:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[base][j]
            ft = func(trial)
            calls += 1
            if ft <= fpop[base]:
                fpop[base] = ft
                pop[base] = trial
                if ft < f_best:
                    f_best = ft
                    x_best = trial.copy()
                    best_idx = base

        # ---- Phase 2: local Nelder-Mead refinement from best point ----
        # adapt initial simplex size to remaining budget
        remaining = budget - calls
        if remaining > 0 and still_going():
            x = x_best.copy()
            f = f_best
            simplex = [x.copy()]
            for j in range(dim):
                p = x.copy()
                step = 0.3 if j == 0 else 0.2
                p[j] += step
                p[j] = np.clip(p[j], -5.0, 5.0)
                simplex.append(p)
            fs = np.empty(dim + 1)
            for i in range(dim + 1):
                fs[i] = func(simplex[i])
                calls += 1
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            it = 0
            max_it = 200
            while it < max_it and calls < budget and still_going():
                order = np.argsort(fs)
                simplex = [simplex[i].copy() for i in order]
                fs = fs[order]
                if fs[0] < f:
                    f = fs[0]
                    x = simplex[0].copy()
                # convergence check
                if fs[-1] - fs[0] < 1e-10:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                calls += 1
                if fr < fs[0]:
                    # expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # contraction
                    if fr < fs[-1]:
                        # outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = func(xc)
                        calls += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            break
                    else:
                        # inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = func(xc)
                        calls += 1
                        if fc <= fs[-1]:
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            break
                it += 1

        return x_best, f_best
# EVOLVE-BLOCK-END
