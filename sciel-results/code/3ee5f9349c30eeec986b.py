"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with stagnation-based restarts and local refinement for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def cma_es(x0, sigma0, budget_remaining):
            """Run CMA-ES from x0 with initial sigma, using up to budget_remaining evaluations."""
            nonlocal calls_made, best_x, best_fx

            d = self.dim
            x = x0.copy()
            fx = float(func(x))
            calls_made += 1
            update_best(x, fx)

            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            cs = np.zeros(d)

            # CMA-ES parameters
            lambda_pop = max(4, int(4 + 3 * np.log(d)))
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (d + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

            B = np.linalg.cholesky(C)
            D = np.ones(d)

            stagnation = 0
            best_local = fx

            for gen in range(budget_remaining):
                if check_time() or calls_made >= self.budget:
                    break

                # Generate lambda candidates
                candidates = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(d)
                    y = x + sigma * (B @ (D * z))
                    candidates.append(y)

                # Evaluate candidates
                fvals = []
                for y in candidates:
                    if check_time() or calls_made >= self.budget:
                        break
                    y_clipped = np.clip(y, -5.0, 5.0)
                    fy = float(func(y_clipped))
                    calls_made += 1
                    fvals.append(fy)
                    if fy < best_local:
                        best_local = fy

                if len(fvals) < lambda_pop:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                x_w = np.zeros(d)
                for i in range(mu):
                    x_w += weights[i] * candidates[order[i]]

                # Update step size (CSA)
                y_old = x
                x_new = x_w
                y = (x_new - x) / sigma
                cs = (1 - c_sigma) * cs + c_sigma * (np.linalg.norm(y) / np.sqrt(2 * d) - 1)
                sigma *= np.exp(c_sigma * d_sigma * cs)
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update evolution paths
                pc = (1 - c_c) * pc + c_c * y
                if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) < (1.4 + 1 / d) * np.sqrt(d):
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (1 - c_c)) * C)
                    for i in range(mu):
                        zi = (candidates[order[i]] - x) / sigma
                        C += c_mu * weights[i] * np.outer(zi, zi)

                # Recalculate B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                x = x_new
                fx = min(fvals)
                update_best(x, fx)

                # Stagnation check
                if fx < best_local - 1e-10:
                    best_local = fx
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 20:
                    break

            return x, fx

        # Phase 1: Multiple restarts with CMA-ES
        num_restarts = 4
        calls_per_restart = (self.budget - 50) // num_restarts

        for r in range(num_restarts):
            if check_time() or calls_made >= self.budget:
                break

            remaining = self.budget - calls_made
            if remaining < 10:
                break

            # Initialize: first restart random, subsequent near best with perturbation
            if best_x is None:
                x0 = rng.uniform(-5.0, 5.0, self.dim)
                sigma0 = 2.0
            else:
                spread = 1.5 + r * 0.5
                x0 = np.clip(best_x + spread * rng.standard_normal(self.dim), -5.0, 5.0)
                sigma0 = 1.0

            x, fx = cma_es(x0, sigma0, calls_per_restart)

        # Phase 2: Local refinement with Nelder-Mead style coordinate search
        if best_x is not None and calls_made < self.budget and not check_time():
            x = best_x.copy()
            fx = best_fx

            # Coordinate-wise local search with adaptive step
            for coord in range(self.dim):
                if check_time() or calls_made >= self.budget:
                    break
                for direction in [1, -1]:
                    if check_time() or calls_made >= self.budget:
                        break
                    step = 0.5
                    for _ in range(15):
                        if check_time() or calls_made >= self.budget:
                            break
                        y = x.copy()
                        y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                        fy = float(func(y))
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            step *= 0.6
                        else:
                            break

            # Fine-tuning with small random steps
            sigma = 0.1
            for i in range(min(30, self.budget - calls_made)):
                if check_time() or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = float(func(y))
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(self.dim)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
