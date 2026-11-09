"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based CMA-ES with Nelder-Mead refinement for Schaffer's F7."""

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
                best_x = np.array(x, dtype=float).copy()
            return fx

        def cma_es(x0, sigma0, budget_left, max_iter=80):
            """Run CMA-ES from x0 with initial sigma0, using up to budget_left calls."""
            nonlocal best_x, best_fx, calls_made
            n = self.dim
            pop = max(4, int(4 + 3 * np.log(n)))
            # CMA-ES parameters
            cc = (4 + n / (4 + n)) / ((n + 4) + (4 + n / (4 + n)))
            cs = (n + 1.5 + 2 / (n + 1)) / ((n + 9) + 2 * (4 + n / (4 + n)))
            c1 = (2 / ((n + 1.3) ** 2 + n)) * (2 - cc - cs)
            cmu = min(0.5, c1 * 2) if c1 > 0 else 0.01
            damps = 1 + 2 * max(0, np.sqrt((pop / 2 - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            for iteration in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break

                if calls_made + pop > self.budget:
                    pop_this = self.budget - calls_made
                    if pop_this < 2:
                        break
                else:
                    pop_this = pop

                # Generate population
                offspring = []
                for i in range(pop_this):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    xi = x_mean + sigma * y
                    fx = call_func(xi)
                    offspring.append((fx, xi))

                if len(offspring) < 2:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                lam = len(offspring)
                mu = max(1, lam // 2)

                # Weights
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights ** 2)

                # Update mean
                x_new_mean = np.zeros(n)
                for i in range(mu):
                    x_new_mean += weights[i] * offspring[i][1]
                step = x_new_mean - x_mean
                x_mean = x_new_mean

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * step
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) / chi_n < 1.4 + 2 / (n + 1)

                if hsig:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * step / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                for i in range(mu):
                    y_i = (offspring[i][1] - x_mean) / sigma
                    C += c1 * np.outer(y_i, y_i)
                    C += (cmu / mueff) * weights[i] * np.outer(y_i, y_i)

                # Re-derive B, D from C periodically
                if iteration % 5 == 0:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.clip(eigenvalues, 1e-12, None)
                    D = np.sqrt(eigenvalues)
                    B = eigenvectors

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Termination: sigma too small
                if sigma < 1e-10:
                    break

            return

        def nelder_mead(x0, step_size, max_iter=100):
            """Nelder-Mead simplex method from x0 with initial step_size."""
            nonlocal best_x, best_fx, calls_made
            n = self.dim

            # Initialize simplex
            simplex = [np.array(x0, dtype=float).copy()]
            for i in range(n):
                xi = np.array(x0, dtype=float).copy()
                xi[i] += step_size
                simplex.append(xi)

            # Evaluate initial simplex
            fx_list = [call_func(simplex[i]) for i in range(n + 1)]

            for iteration in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]

                # Check convergence
                if abs(fx_list[0] - fx_list[-1]) < 1e-10 * max(1.0, abs(fx_list[0])):
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                fx_reflect = call_func(x_reflect)

                if fx_list[0] <= fx_reflect < fx_list[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fx_list[-1] = fx_reflect
                elif fx_reflect < fx_list[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    fx_expand = call_func(x_expand)
                    if fx_expand < fx_reflect:
                        simplex[-1] = x_expand
                        fx_list[-1] = fx_expand
                    else:
                        simplex[-1] = x_reflect
                        fx_list[-1] = fx_reflect
                else:
                    # Contraction
                    if fx_reflect < fx_list[-1]:
                        # Outside contraction
                        rho = 0.5
                        x_contract = centroid + rho * (x_reflect - centroid)
                        fx_contract = call_func(x_contract)
                        if fx_contract <= fx_reflect:
                            simplex[-1] = x_contract
                            fx_list[-1] = fx_contract
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fx_list[i] = call_func(simplex[i])
                    else:
                        # Inside contraction
                        rho = 0.5
                        x_contract = centroid - rho * (centroid - simplex[-1])
                        fx_contract = call_func(x_contract)
                        if fx_contract < fx_list[-1]:
                            simplex[-1] = x_contract
                            fx_list[-1] = fx_contract
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fx_list[i] = call_func(simplex[i])

                # Sort again
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]

            return

        # Phase 1: Diverse initial sampling
        n_initial = min(20, self.budget)
        x0 = np.zeros(self.dim)
        fx0 = call_func(x0)
        best_fx = fx0
        best_x = x0.copy()

        for i in range(1, n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Island-based CMA-ES (3 independent runs)
        n_islands = 3
        budget_per_island = (self.budget - calls_made) // (n_islands + 1)  # Reserve for NM

        for r in range(n_islands):
            if calls_made >= self.budget or time.time() > deadline:
                break

            if r == 0:
                x_start = best_x.copy()
                sigma_start = 1.0
            else:
                # Restart from perturbed best or random
                if rng.random() < 0.5:
                    x_start = best_x + 1.5 * rng.standard_normal(self.dim)
                else:
                    x_start = rng.uniform(-4, 4, self.dim)
                x_start = np.clip(x_start, -5, 5)
                sigma_start = 1.0

            cma_es(x_start, sigma_start, budget_per_island)

        # Phase 3: Nelder-Mead refinement on best point
        if calls_made < self.budget - 5 and time.time() < deadline:
            nelder_mead(best_x.copy(), 0.5, max_iter=50)

        # Phase 4: Final fine-tuning with small sigma CMA-ES
        if calls_made < self.budget - 10 and time.time() < deadline:
            cma_es(best_x.copy(), 0.05, self.budget - calls_made, max_iter=30)

        return best_x, best_fx
# EVOLVE-BLOCK-END
