"""Rewritten optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining Differential Evolution (DE) for global exploration
    with Nelder-Mead for local refinement on Rastrigin.
    Includes a wall-clock deadline guard (300s limit, return best ~10s before).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = x_clip.copy()
            return f_val

        # --- Phase 1: Differential Evolution for global exploration ---
        # Population size
        pop_size = 15
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        # Budget allocation: use ~60% for DE, ~40% for local refinement
        de_budget = int(budget * 0.6)
        de_calls = 0

        # Initialize population: mix of random and grid-based points
        # Rastrigin has local minima at integers, so include integer-like points
        pop = []
        # Add some structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
        ]
        for sp in structured:
            if len(pop) < pop_size:
                pop.append(sp.copy())
        # Fill rest with random points
        while len(pop) < pop_size:
            pop.append(rng.uniform(-5, 5, dim))

        pop = np.array(pop)
        pop_f = np.array([safe_call(p) for p in pop])

        if np.any(pop_f is None):
            return best_x, best_fx

        # DE loop
        while de_calls < de_budget and calls_made < budget and time.time() < deadline:
            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_fx:
                best_fx = pop_f[0]
                best_x = pop[0].copy()

            # Generate offspring
            new_pop = pop.copy()
            new_pop_f = pop_f.copy()

            for i in range(pop_size):
                if de_calls + (pop_size - i) > de_budget:
                    break

                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break

                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover: binomial
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                trial = np.clip(trial, -5, 5)
                ft = safe_call(trial)

                if ft is None:
                    break

                de_calls += 1

                # Selection
                if ft <= pop_f[i]:
                    new_pop[i] = trial
                    new_pop_f[i] = ft

            pop = new_pop
            pop_f = new_pop_f

        # --- Phase 2: Nelder-Mead local refinement ---
        # Use the best point from DE as starting point
        # Also try a few other good points from the population
        nm_budget = budget - calls_made

        # Get top 3 points from DE population
        order = np.argsort(pop_f)
        start_points = [pop[order[k]].copy() for k in range(min(3, len(order)))]

        # Run Nelder-Mead on each starting point, distributing remaining budget
        nm_per_start = max(1, nm_budget // len(start_points))

        for sp in start_points:
            if calls_made >= budget or time.time() > deadline:
                break

            # Initialize simplex around sp
            simplex = [sp.copy()]
            fs = [safe_call(sp)]
            if fs[0] is None:
                break

            for i in range(dim):
                xi = sp.copy()
                # Small initial step, scaled by local landscape
                xi[i] += 0.1
                simplex.append(xi)
                fi = safe_call(xi)
                if fi is None:
                    break
                fs.append(fi)

            if calls_made >= budget or time.time() > deadline:
                break

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while calls_made < budget and time.time() < deadline:
                # Sort simplex
                order_s = np.argsort(fs)
                simplex = [simplex[i] for i in order_s]
                fs = [fs[i] for i in order_s]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        break
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
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if calls_made >= budget or time.time() > deadline:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
