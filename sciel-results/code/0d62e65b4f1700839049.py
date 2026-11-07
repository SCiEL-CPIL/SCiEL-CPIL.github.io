import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        def check_time():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or not check_time():
                return None
            val = func(x)
            calls_made += 1
            return val

        def cma_es(x0, f0, sigma, max_evals):
            """Run CMA-ES from starting point x0 with initial sigma."""
            nonlocal best_x, best_f, calls_made

            N = dim
            lam = 10  # population size
            mu = lam // 2  # number of parents
            weights = np.log(np.arange(lam, 0, -1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights[:mu]**2)

            # CMA-ES parameters
            cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
            cs = (mu_eff + 2) / (N + mu_eff + 5)
            c1 = 2 / ((N + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2)**2 + mu_eff))

            C = np.eye(N)
            pc = np.zeros(N)
            ps = np.zeros(N)
            chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))

            x_mean = x0.copy()
            f_mean = f0

            evals = 0
            stagnation = 0
            best_local_x = x0.copy()
            best_local_f = f0

            while evals < max_evals and calls_made < budget and check_time():
                # Generate population
                Y = rng.standard_normal((lam, N))
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.abs(D))
                C_sqrt_inv = B @ np.diag(1.0 / D) @ B.T
                Z = Y @ C_sqrt_inv
                population = x_mean + sigma * Z

                # Evaluate population
                f_values = np.full(lam, np.inf)
                for i in range(lam):
                    x_i = np.clip(population[i], -5, 5)
                    f_i = safe_call(x_i)
                    if f_i is None:
                        break
                    f_values[i] = f_i
                    evals += 1
                    if f_i < best_local_f:
                        best_local_f = f_i
                        best_local_x = x_i.copy()
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()

                if evals >= max_evals or calls_made >= budget:
                    break

                # Sort by fitness
                order = np.argsort(f_values)
                Y_selected = Z[order[:mu]]
                f_selected = f_values[order[:mu]]

                # Update mean
                x_mean_new = x_mean + sum(weights[i] * Y_selected[i] for i in range(mu))

                # Update covariance matrix
                v = x_mean_new - x_mean
                pc = (1 - cc) * pc + cc * (v / sigma)
                ps = (1 - cs) * ps + cs * np.sqrt(cc * (2 - cc) / (N + 1)) * (pc / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chiN < 1.4 + 2 / (N + 1))
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(weights[i] * np.outer(Y_selected[i], Y_selected[i]) for i in range(mu))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(N)

                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x_mean = x_mean_new

                # Check for stagnation
                if best_local_f < f0 - 1e-8:
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 20:
                    break

            return best_local_x, best_local_f

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Initial sampling to find good starting points
        n_initial = min(50, budget // 10)
        initial_points = []
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, size=dim)
            f = safe_call(x)
            if f is None:
                break
            initial_points.append((f, x))

        if not initial_points:
            return np.zeros(dim), 0.0

        initial_points.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top candidates
        n_starts = min(5, len(initial_points))
        remaining_budget = budget - calls_made
        budget_per_start = max(50, remaining_budget // n_starts)

        for i in range(n_starts):
            if calls_made >= budget or not check_time():
                break

            f_start, x_start = initial_points[i]
            sigma_start = 1.0

            best_local_x, best_local_f = cma_es(x_start, f_start, sigma_start, budget_per_start)

            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()

        if best_x is None:
            best_x = initial_points[0][1]
            best_f = initial_points[0][0]

        return best_x, best_f
