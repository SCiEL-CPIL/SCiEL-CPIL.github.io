"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead based optimizer for Schaffer's F7 with hidden rotation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, step_size, max_iter=100):
            """Nelder-Mead simplex method from x0 with initial step_size."""
            nonlocal calls_made, best_x, best_fx
            n = self.dim
            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += step_size
                simplex.append(xi)

            # Evaluate initial simplex
            for i in range(n + 1):
                if calls_made >= self.budget or time.time() > deadline:
                    return
                simplex[i] = simplex[i].copy()
                simplex[i].append(call_func(simplex[i][:n]))

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            for iteration in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda s: s[-1])

                # Check for convergence
                if abs(simplex[-1][-1] - simplex[0][-1]) < 1e-10:
                    break

                x_worst = simplex[-1][:n]
                x_second_worst = simplex[-2][:n]
                x_best = simplex[0][:n]

                # Centroid of all but worst point
                x_centroid = np.mean([s[:n] for s in simplex[:-1]], axis=0)

                # Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                fx_reflect = call_func(x_reflect)

                if fx_reflect < simplex[0][-1]:
                    # Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                    fx_expand = call_func(x_expand)
                    if fx_expand < fx_reflect:
                        simplex[-1] = np.append(x_expand, fx_expand)
                    else:
                        simplex[-1] = np.append(x_reflect, fx_reflect)
                elif fx_reflect < simplex[-2][-1]:
                    # Accept reflection
                    simplex[-1] = np.append(x_reflect, fx_reflect)
                else:
                    # Contraction
                    if fx_reflect < simplex[-1][-1]:
                        # Outside contraction
                        x_contract = x_centroid + rho * (x_reflect - x_centroid)
                    else:
                        # Inside contraction
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    fx_contract = call_func(x_contract)

                    if fx_contract < min(fx_reflect, simplex[-1][-1]):
                        simplex[-1] = np.append(x_contract, fx_contract)
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = np.append(x_best + sigma_shrink * (simplex[i][:n] - x_best), simplex[i][-1])
                            # Re-evaluate
                            if calls_made >= self.budget or time.time() > deadline:
                                return
                            simplex[i][-1] = call_func(simplex[i][:n])

            return

        # Phase 1: Diverse initial sampling to find good basins
        n_initial = min(20, self.budget)
        # Try origin first
        x0 = np.zeros(self.dim)
        fx0 = call_func(x0)
        best_fx = fx0
        best_x = x0.copy()

        # Random sampling
        for i in range(1, n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Multi-restart Nelder-Mead
        # Allocate budget across restarts
        n_restarts = 6
        budget_per_restart = max(30, (self.budget - calls_made) // n_restarts)

        for r in range(n_restarts):
            if calls_made >= self.budget or time.time() > deadline:
                break

            if r == 0:
                # Start from best found
                x_start = best_x.copy()
                step_size = 1.0
            else:
                # Restart: perturb from best or random
                if rng.random() < 0.5:
                    x_start = best_x + 2.0 * rng.standard_normal(self.dim)
                else:
                    x_start = rng.uniform(-4, 4, self.dim)
                x_start = np.clip(x_start, -5, 5)
                step_size = 1.0

            # Limit iterations to stay within budget
            max_iter = min(150, (self.budget - calls_made) // (self.dim + 1))
            nelder_mead(x_start, step_size, max_iter=max_iter)

        # Phase 3: Final fine-tuning with small step size
        if calls_made < self.budget - 10 and time.time() < deadline:
            max_iter = min(100, (self.budget - calls_made) // (self.dim + 1))
            nelder_mead(best_x.copy(), 0.1, max_iter=max_iter)

        return best_x, best_fx
# EVOLVE-BLOCK-END
