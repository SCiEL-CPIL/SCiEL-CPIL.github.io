"""Multi-scale coordinate descent + CMA-ES hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale coordinate descent for early progress + CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_coordinate_descent(x0, max_calls):
            """Multi-scale coordinate descent with geometrically shrinking steps."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            x = np.array(x0, dtype=np.float64)
            f = evaluate(x)
            if calls_made >= budget:
                return

            # Start with large steps, shrink geometrically
            step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break

                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial < f:
                        x = trial.copy()
                        f = f_trial
                    else:
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f:
                            x = trial.copy()
                            f = f_trial

            # Continue with smaller steps if budget remains
            for step in [0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]:
                if calls_made >= budget or time.time() > deadline:
                    break

                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial < f:
                        x = trial.copy()
                        f = f_trial
                    else:
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f:
                            x = trial.copy()
                            f = f_trial

        def run_cmaes(x0, max_calls):
            """Run CMA-ES style optimization using scipy's minimize with Nelder-Mead as fallback."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 5 or time.time() > deadline:
                return

            local_x = np.array(x0, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        def de_search(max_calls, pop_size=12, F_mut=0.8, CR=0.5):
            """Differential evolution for global exploration."""
            nonlocal calls_made, x_best, f_best
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            pop_size = min(pop_size, max(8, max_calls // 15))

            pop = []
            if x_best is not None:
                pop.append((x_best.copy(), f_best))

            while len(pop) < pop_size:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                f = evaluate(x)
                pop.append((x, f))

            if not pop:
                return

            pop.sort(key=lambda item: item[1])

            while calls_made < max_calls and time.time() < deadline:
                x_best_pop = pop[0][0]
                new_pop = []

                for i in range(pop_size):
                    if calls_made >= max_calls or time.time() > deadline:
                        new_pop.append(pop[i])
                        continue

                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 3:
                        new_pop.append(pop[i])
                        continue
                    r1, r2, r3 = rng.choice(indices, 3, replace=False)

                    v = x_best_pop + F_mut * (pop[r1][0] - pop[r2][0])

                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][0][j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial <= pop[i][1]:
                        new_pop.append((trial, f_trial))
                    else:
                        new_pop.append(pop[i])

                pop = new_pop
                pop.sort(key=lambda item: item[1])

            if pop and pop[0][1] < f_best:
                f_best = pop[0][1]
                x_best = pop[0][0].copy()

        def lattice_init(max_calls):
            """Probe integer lattice points to find the correct basin for f4."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= max_calls or time.time() > deadline:
                return

            rng = np.random.default_rng(42)
            lattice_budget = min(max_calls, int(budget * 0.1))

            for _ in range(lattice_budget):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
                evaluate(x_int)

            if x_best is not None and calls_made < budget - 10:
                remaining = budget - calls_made
                nm_calls = min(10, remaining)
                run_cmaes(x_best.copy(), nm_calls)

        # === Main Strategy ===

        # Phase 1: Lattice Search (10% budget) - find correct basin
        lattice_calls = int(budget * 0.1)
        if time.time() < deadline:
            lattice_init(lattice_calls)

        # Phase 2: DE Global Search (40% budget)
        de_calls = int(budget * 0.4)
        if time.time() < deadline:
            de_search(de_calls, pop_size=12)

        # Phase 3: Multi-scale Coordinate Descent (25% budget)
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            cd_calls = min(remaining, int(budget * 0.25))
            run_coordinate_descent(x_best.copy(), cd_calls)

        # Phase 4: Nelder-Mead Refinement (15% budget)
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, int(budget * 0.15))
            run_cmaes(x_best.copy(), nm_calls)

        # Phase 5: Final Polish - alternate CD and NM with remaining budget
        remaining = budget - calls_made
        while remaining >= 10 and x_best is not None and time.time() < deadline:
            cd_calls = min(remaining // 2, 15)
            if cd_calls >= 4:
                run_coordinate_descent(x_best.copy(), cd_calls)
                remaining = budget - calls_made
                if remaining < 10:
                    break

            nm_calls = min(remaining // 2, 15)
            if nm_calls >= 8:
                run_cmaes(x_best.copy(), nm_calls)
                remaining = budget - calls_made

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
