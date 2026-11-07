"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Global search with adaptive (mu+lambda)-ES, followed by Nelder-Mead polish.
    Designed for Lunacek bi-Rastrigin (f24) with hidden shift/rotation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        eval_count = 0

        def eval_func(x):
            nonlocal eval_count
            eval_count += 1
            return func(np.clip(x, -5.0, 5.0))

        rng = np.random.default_rng(42)

        # --- Phase 1: Global exploration ---
        mu = 4
        lam = 12
        sigma = 1.5
        x_best = rng.uniform(-4.0, 4.0, dim)
        f_best = eval_func(x_best)

        # Maintain a population of the best points
        population = [(x_best.copy(), f_best)]

        while eval_count < int(budget * 0.75) and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lam):
                step = sigma * rng.standard_normal(dim)
                # Occasionally do a larger jump
                if rng.random() < 0.08:
                    step *= 2.5
                y = x_best + step
                offspring.append(y)

            # Evaluate offspring and collect fitnesses
            offspring_fits = []
            for y in offspring:
                if time.time() > deadline:
                    break
                fy = eval_func(y)
                offspring_fits.append(fy)
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()

            # Update population: keep best mu from {current population + offspring}
            candidates = [(x, f) for x, f in population]
            for i, y in enumerate(offspring):
                if i < len(offspring_fits):
                    candidates.append((y, offspring_fits[i]))

            # Sort by fitness
            candidates.sort(key=lambda t: t[1])
            population = candidates[:mu]

            # Adapt sigma: shrink slowly
            sigma *= 0.98
            if sigma < 0.05:
                sigma = 0.05

        # --- Phase 2: Local polish with Nelder-Mead ---
        remaining = budget - eval_count
        if remaining > 10 and time.time() < deadline:
            x_nm = x_best.copy()
            f_nm = f_best

            # Build initial simplex
            initial_step = 0.3
            simplex = [x_nm.copy()]
            simplex_f = [f_nm]
            for i in range(dim):
                point = x_nm.copy()
                point[i] += initial_step
                simplex.append(point)
                if time.time() > deadline:
                    break
                simplex_f.append(eval_func(point))

            # Nelder-Mead iterations
            max_nm_iters = min(remaining - (dim + 1), 150)
            for _ in range(max_nm_iters):
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                f_ref = eval_func(x_ref)

                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    f_exp = eval_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst - centroid)
                    f_con = eval_func(x_con)
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])

                # Update best
                min_idx = int(np.argmin(simplex_f))
                if simplex_f[min_idx] < f_best:
                    f_best = simplex_f[min_idx]
                    x_best = simplex[min_idx].copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
