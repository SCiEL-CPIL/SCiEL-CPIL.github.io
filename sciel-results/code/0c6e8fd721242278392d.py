"""Differential Evolution + Nelder-Mead optimizer for BBOB f19 (Griewank-Rosenbrock)."""
import numpy as np
import time
import math


class Optimizer:
    """DE-based optimizer with adaptive mutation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # Nelder-Mead local polish
        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            n = dim
            simplex = [x0.copy()]
            f0 = safe_eval(x0)
            if f0 is None:
                return
            simplex_f = [f0]

            # Build initial simplex
            for i in range(n):
                xi = x0.copy()
                step = 0.25 if abs(x0[i]) < 1e-10 else 0.25 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = safe_eval(xi)
                if fi is None:
                    return
                simplex.append(xi)
                simplex_f.append(fi)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for gen in range(max_calls // (n + 1) + 5):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_eval(xr)
                if fr is None:
                    break

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_eval(xe)
                    if fe is None:
                        break
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_eval(xc)
                    if fc is None:
                        break
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_eval(simplex[i])
                            if fi is None:
                                return
                            simplex_f[i] = fi

                        if calls_made >= budget or time.time() >= deadline:
                            break

                # Check convergence
                if len(set([round(f, 12) for f in simplex_f])) == 1:
                    break

            return

        # Differential Evolution
        def differential_evolution(pop_size, max_calls):
            nonlocal calls_made, best_x, best_f

            if calls_made >= budget:
                return

            # Initialize population
            pop = []
            pop_f = []
            for i in range(pop_size):
                if calls_made >= budget or time.time() >= deadline:
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                fx = safe_eval(x)
                if fx is not None:
                    pop.append(x)
                    pop_f.append(fx)

            if len(pop) < 4:
                return

            pop = np.array(pop)
            pop_f = np.array(pop_f)

            # Adaptive DE with CR and F
            cr = 0.7
            f_low, f_high = 0.4, 0.9
            f_current = 0.7
            no_improve = 0
            max_no_improve = 20

            for gen in range(max_calls // pop_size + 5):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Sort population
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                gen_best = pop_f[0]

                for i in range(len(pop)):
                    if calls_made >= budget or time.time() >= deadline:
                        break

                    # DE/rand/1 mutation
                    idxs = [j for j in range(len(pop)) if j != i]
                    a, b, c = rng.choice(idxs, size=3, replace=False)

                    trial = pop[a] + f_current * (pop[b] - pop[c])
                    trial = np.clip(trial, -5, 5)

                    ft = safe_eval(trial)
                    if ft is None:
                        break

                    # Selection
                    if ft <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = ft

                # Update best
                if pop_f[0] < gen_best - 1e-10:
                    no_improve = 0
                else:
                    no_improve += 1

                # Adapt F
                if no_improve > max_no_improve:
                    f_current = rng.uniform(f_low, f_high)
                    cr = rng.uniform(0.5, 0.9)
                    no_improve = 0

                # Restart if stuck for too long
                if no_improve > 60:
                    # Perturb worst individuals
                    for i in range(len(pop) // 2):
                        worst_idx = len(pop) - 1 - i
                        pop[worst_idx] = best_x + rng.standard_normal(dim) * 1.0
                        pop[worst_idx] = np.clip(pop[worst_idx], -5, 5)
                        ft = safe_eval(pop[worst_idx])
                        if ft is not None:
                            pop_f[worst_idx] = ft
                    no_improve = 0

                if calls_made >= budget or time.time() >= deadline:
                    break

            return

        # Phase 1: Initial diverse evaluation
        n_initial = 10
        for i in range(n_initial):
            if calls_made >= budget or time.time() >= deadline:
                break
            x0 = rng.uniform(-4.5, 4.5, dim)
            safe_eval(x0)

        # Also evaluate origin and a few structured points
        for x0 in [np.zeros(dim), np.ones(dim) * 0.5, -np.ones(dim) * 0.5]:
            if calls_made >= budget or time.time() >= deadline:
                break
            safe_eval(x0)

        # Phase 2: Main DE search
        if calls_made < budget:
            remaining = budget - calls_made
            # Use most of the budget for DE
            de_calls = int(remaining * 0.75)
            differential_evolution(pop_size=12, max_calls=de_calls)

        # Phase 3: Nelder-Mead polish from best point
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            nm_calls = min(remaining, 100)
            nelder_mead(best_x.copy(), nm_calls)

        # Phase 4: Additional DE from perturbed best if budget remains
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                # Restart DE from neighborhood of best
                pop_size = 8
                for i in range(pop_size):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    x = best_x + rng.standard_normal(dim) * 0.8
                    x = np.clip(x, -5, 5)
                    safe_eval(x)

                if calls_made < budget:
                    differential_evolution(pop_size=pop_size, max_calls=budget - calls_made)

        # Phase 5: Final Nelder-Mead polish
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                nelder_mead(best_x.copy(), min(remaining, 50))

        return best_x, best_f
