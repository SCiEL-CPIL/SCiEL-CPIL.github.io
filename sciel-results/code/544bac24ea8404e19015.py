"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with Nelder-Mead polish for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize population for DE
        pop_size = 20
        # Spread initial points including some structured ones
        pop = np.zeros((pop_size, dim))
        # First few points: grid-like spread
        pop[0] = np.zeros(dim)
        for i in range(1, min(4, pop_size)):
            pop[i] = rng.uniform(-4, 4, dim)
        for i in range(4, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)

        fpop = np.array([call_func(p) for p in pop])
        best_idx = np.argmin(fpop)
        x_best = pop[best_idx].copy()
        f_best = fpop[best_idx]

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Main DE loop
        while calls_made < budget:
            if time.time() > deadline - 60.0:
                break

            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 60.0:
                    break

                # Select 3 distinct individuals
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break

                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                ftrial = call_func(trial)

                if ftrial <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = ftrial
                    if ftrial < f_best:
                        x_best = trial.copy()
                        f_best = ftrial

        # Nelder-Mead polish on the best solution
        if calls_made < budget and time.time() < deadline - 10.0:
            nm_budget = min(budget - calls_made, 100)
            if nm_budget > 10:
                x_nm = x_best.copy()
                f_nm = f_best

                # Initial simplex
                simplex = [x_nm.copy()]
                for j in range(dim):
                    pt = x_nm.copy()
                    step = 0.5 if x_nm[j] < 0 else -0.5
                    pt[j] += step
                    pt = np.clip(pt, -5, 5)
                    simplex.append(pt)

                f_simplex = np.array([call_func(s) for s in simplex])
                simplex = np.array(simplex)

                max_nm_iter = 50
                for _ in range(max_nm_iter):
                    if calls_made >= budget or time.time() > deadline - 5.0:
                        break

                    # Sort by function value
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]

                    f_worst = f_simplex[-1]
                    f_second = f_simplex[-2]
                    f_best_nm = f_simplex[0]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    alpha = 1.0
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = call_func(x_r)

                    if f_r < f_simplex[0]:
                        # Expansion
                        gamma = 2.0
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        f_e = call_func(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_second:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        # Contraction
                        beta = 0.5
                        x_c = centroid + beta * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = call_func(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for k in range(1, dim + 1):
                                simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                simplex[k] = np.clip(simplex[k], -5, 5)
                                f_simplex[k] = call_func(simplex[k])

                    # Update best
                    idx_best = np.argmin(f_simplex)
                    if f_simplex[idx_best] < f_nm:
                        f_nm = f_simplex[idx_best]
                        x_nm = simplex[idx_best].copy()

                if f_nm < f_best:
                    x_best = x_nm
                    f_best = f_nm

        return x_best, f_best
# EVOLVE-BLOCK-END
