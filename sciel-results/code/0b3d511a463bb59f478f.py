"""Optimizer for BBOB-style functions using a robust multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-strategy optimizer for 5D BBOB functions.
    Combines:
    1. Dense initial sampling
    2. Differential Evolution for global exploration
    3. L-BFGS-B local refinement from multiple points
    4. (1+1)-ES fine-tuning
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            val = float(func(x))
            if not np.isfinite(val):
                return 1e10
            return val

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = np.array(x, dtype=float)
                best_f = f

        def check_time():
            return time.time() < deadline and calls[0] < budget

        # Phase 1: Dense initial sampling
        # Use a grid of points plus random samples
        n_initial = min(30, budget)
        points = []
        # Center
        points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                p = np.zeros(dim)
                p[i] = val
                points.append(p)
        # Random points
        while len(points) < n_initial:
            points.append(rng.uniform(-5, 5, dim))

        for x in points:
            if not check_time():
                break
            x = np.clip(x, -5, 5)
            try:
                f = wrapped(x)
                update_best(x, f)
            except Exception:
                pass

        # Phase 2: Differential Evolution
        def differential_evolution(start_best_x, pop_size=20, max_gen=20, F=0.8, CR=0.9):
            nonlocal calls, best_x, best_f, deadline
            n = dim
            # Initialize population
            pop = [start_best_x.copy()]
            while len(pop) < pop_size:
                x = start_best_x + rng.standard_normal(n) * 3.0
                x = np.clip(x, -5, 5)
                pop.append(x)

            pop_f = [np.inf] * pop_size
            for i, x in enumerate(pop):
                if not check_time():
                    return
                try:
                    pop_f[i] = wrapped(x)
                except Exception:
                    pop_f[i] = 1e10
                update_best(x, pop_f[i])

            for gen in range(max_gen):
                if not check_time():
                    break

                new_pop = []
                new_pop_f = []
                for i in range(pop_size):
                    # DE/rand/1
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, -5, 5)

                    # Crossover
                    trial = mutant.copy()
                    for j in range(n):
                        if rng.random() < CR or j == rng.randint(n):
                            trial[j] = mutant[j]
                        else:
                            trial[j] = pop[i][j]

                    try:
                        f_trial = wrapped(trial)
                    except Exception:
                        f_trial = 1e10

                    if f_trial <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(f_trial)
                        update_best(trial, f_trial)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])

                pop = new_pop
                pop_f = new_pop_f

                # Check for stagnation
                if gen > 5:
                    recent_best = min(pop_f)
                    if abs(recent_best - best_f) < 1e-10:
                        # Check if we've been stuck for a while
                        if gen > 10:
                            break

        if best_x is not None and check_time():
            differential_evolution(best_x.copy())

        # Phase 3: L-BFGS-B from multiple starting points
        def local_refine(x0, max_iter=50):
            nonlocal calls, best_x, best_f, deadline

            def obj(x):
                if not check_time():
                    return 1e10
                x = np.clip(x, -5, 5)
                try:
                    return wrapped(x)
                except Exception:
                    return 1e10

            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': max_iter, 'maxfun': 30})
                if res.success or res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        if best_x is not None and check_time():
            # Refine from the best point
            local_refine(best_x.copy())

            # Also try a few perturbed starts
            for _ in range(3):
                if not check_time():
                    break
                x0 = best_x + rng.standard_normal(dim) * 1.0
                x0 = np.clip(x0, -5, 5)
                local_refine(x0)

        # Phase 4: (1+1)-ES fine-tuning
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            max_no_improve = 30

            while check_time():
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                if no_improve > max_no_improve:
                    break
                if sigma < 1e-10:
                    break

        # Phase 5: Final polish with small L-BFGS-B
        if best_x is not None and check_time() and calls[0] < budget - 5:
            def final_obj(x):
                if not check_time():
                    return 1e10
                x = np.clip(x, -5, 5)
                try:
                    return wrapped(x)
                except Exception:
                    return 1e10

            try:
                res = minimize(final_obj, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 20, 'maxfun': 10,
                                        'ftol': 1e-15, 'gtol': 1e-15})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
