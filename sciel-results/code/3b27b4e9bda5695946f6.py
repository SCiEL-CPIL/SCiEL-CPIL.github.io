"""Direct solver for BBOB f9 (rotated Rosenbrock) using DE global search + BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE global search for rotated Rosenbrock valley, followed by BFGS polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution for global search
        # DE is robust to unknown rotations/shifts due to population diversity
        pop_size = 30
        max_de_iters = 50  # 30 * 50 = 1500 max, but we cap at budget

        # Initialize population
        population = rng.uniform(-4.0, 4.0, size=(pop_size, dim))
        pop_f = np.array([counted_func(p) for p in population])

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        de_iter = 0
        while de_iter < max_de_iters:
            if call_count[0] >= budget - 100:  # leave budget for polish
                break
            if time.time() > deadline - 60.0:  # leave time for polish
                break

            # Find best in population
            best_idx = np.argmin(pop_f)
            best_val = pop_f[best_idx]

            for i in range(pop_size):
                if call_count[0] >= budget - 100:
                    break
                if time.time() > deadline - 60.0:
                    break

                # Select 3 distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5.0, 5.0)

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                trial = np.clip(trial, -5.0, 5.0)

                # Selection
                trial_f = counted_func(trial)
                if trial_f <= pop_f[i]:
                    population[i] = trial
                    pop_f[i] = trial_f

            de_iter += 1

        # Get best from DE
        de_best_idx = np.argmin(pop_f)
        de_best_x = population[de_best_idx].copy()
        de_best_f = pop_f[de_best_idx]

        if de_best_f < best_f[0]:
            best_f[0] = de_best_f
            best_x[0] = de_best_x.copy()

        # Phase 2: L-BFGS-B polish from DE best
        if best_x[0] is not None and call_count[0] < budget - 20 and time.time() < deadline - 20.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: BFGS polish with dense Hessian for final precision
        if best_x[0] is not None and call_count[0] < budget - 10 and time.time() < deadline - 10.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-12, 'disp': False}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B with tightest tolerances
        if best_x[0] is not None and call_count[0] < budget - 5 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-15}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = counted_func(best_x[0])

        return best_x[0], best_f[0]
