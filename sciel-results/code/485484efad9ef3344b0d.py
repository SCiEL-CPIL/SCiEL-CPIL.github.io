import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-strategy optimizer for BBOB-style functions in 5D.
    Combines:
    - Structured initial sampling
    - CMA-ES (via scipy's minimize or manual)
    - Differential Evolution
    - L-BFGS-B polish
    - Multi-start with diverse initializations
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # Shared archive of best solutions found
        archive = []  # List of (f, x) tuples, sorted by f

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
                # Add to archive (keep top 15)
                archive.append((val, x.copy()))
                archive.sort(key=lambda t: t[0])
                if len(archive) > 15:
                    archive.pop()
            return val

        def polish(x_start, maxfev=30):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': maxfev,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Initial sampling with structured + random points
        # Use 8 diverse starts: zero, corners, random, perturbed
        init_points = [
            np.zeros(dim),
            np.full(dim, -4.0),
            np.full(dim, 4.0),
            np.full(dim, -3.0),
            np.full(dim, 3.0),
        ]
        # Add random points
        for i in range(3):
            init_points.append(rng.uniform(-4.5, 4.5, dim))

        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Quick polish from initial best
        polish(best_x, maxfev=max(10, (budget - calls) // 15))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES-like search using scipy's Nelder-Mead or Powell
        # Try Powell's method which is derivative-free and robust
        def run_powell(x_start, maxiter=30):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'xtol': 1e-10,
                        'ftol': 1e-10,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Run Powell from multiple starting points
        remaining = budget - calls
        if remaining > 30:
            n_powell = min(5, remaining // 20)
            for i in range(n_powell):
                if calls >= budget or time.time() > deadline:
                    break
                if i == 0:
                    x_start = best_x.copy()
                elif i < len(archive):
                    _, x_arch = archive[i % len(archive)]
                    x_start = x_arch + rng.normal(0, 0.3, dim)
                else:
                    x_start = best_x + rng.normal(0, 1.0, dim)
                x_start = np.clip(x_start, -5, 5)
                run_powell(x_start, maxiter=min(30, (budget - calls) // max(1, n_powell - i)))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Differential Evolution for global exploration
        def run_de(pop_size, max_generations, crossover_rate=0.7, mutation_scale=1.0):
            """Run differential evolution with given parameters."""
            nonlocal calls, best_f, best_x, archive

            if calls >= budget or time.time() > deadline:
                return

            # Initialize population from archive and random points
            pop = []
            for i in range(pop_size):
                if i < len(archive) and i % 2 == 0:
                    # Use archived solution with perturbation
                    _, x_arch = archive[i % len(archive)]
                    x = x_arch + rng.normal(0, 0.5, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                pop.append(x)

            # Evaluate initial population
            pop_f = []
            for x in pop:
                if calls >= budget or time.time() > deadline:
                    return
                f = wrapped_func(x)
                pop_f.append(f)

            if len(pop_f) < pop_size:
                return

            # DE parameters
            F = mutation_scale  # Mutation scale

            for gen in range(max_generations):
                if calls >= budget or time.time() > deadline:
                    break

                new_pop = []
                new_pop_f = []

                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Select three distinct random individuals
                    indices = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(indices, 3, replace=False)

                    # Mutation
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, -5, 5)

                    # Crossover
                    crossover = rng.random(dim) < crossover_rate
                    if not crossover.any():
                        crossover[rng.randint(dim)] = True

                    trial = np.where(crossover, mutant, pop[i])
                    trial = np.clip(trial, -5, 5)

                    # Evaluate trial
                    f_trial = wrapped_func(trial)

                    # Selection
                    if f_trial <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(f_trial)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])

                if len(new_pop_f) < pop_size:
                    break

                pop = new_pop
                pop_f = new_pop_f

                # Adaptive crossover rate
                improvement = sum(1 for i in range(pop_size) if new_pop_f[i] < pop_f[i])
                if improvement > pop_size * 0.3:
                    crossover_rate = min(0.9, crossover_rate * 1.1)
                elif improvement < pop_size * 0.1:
                    crossover_rate = max(0.5, crossover_rate * 0.9)

            # Update archive with best from this DE run
            best_idx = np.argmin(pop_f)
            if pop_f[best_idx] < best_f:
                best_f = pop_f[best_idx]
                best_x = pop[best_idx].copy()
                archive.append((pop_f[best_idx], pop[best_idx].copy()))
                archive.sort(key=lambda t: t[0])
                if len(archive) > 15:
                    archive.pop()

        # Run DE with moderate population
        remaining = budget - calls
        if remaining > 40:
            run_de(pop_size=20, max_generations=8, crossover_rate=0.7, mutation_scale=1.2)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Multiple L-BFGS-B polish runs from diverse points
        if best_x is not None and calls < budget and time.time() < deadline:
            n_local = min(8, (budget - calls) // 10)
            for ls in range(n_local):
                if calls >= budget or time.time() > deadline:
                    break

                if ls == 0:
                    x_start = best_x.copy()
                elif ls < len(archive):
                    _, x_arch = archive[ls % len(archive)]
                    x_start = x_arch + rng.normal(0, 0.2, dim)
                else:
                    x_start = best_x + rng.normal(0, 0.5, dim)

                x_start = np.clip(x_start, -5, 5)
                remaining = budget - calls
                maxfev = max(5, remaining // max(1, n_local - ls))
                polish(x_start, maxfev=maxfev)

        # Phase 5: Final high-precision polish
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'maxfev': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-12,
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
