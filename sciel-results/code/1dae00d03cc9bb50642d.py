import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        n_calls = 0

        def safe_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget or time.time() > deadline:
                raise RuntimeError("Budget or time exceeded")
            n_calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        def cma_es_run(x0, sigma0, max_evals, seed_offset=0):
            """Run CMA-ES from x0 with given budget, return (best_x, best_f, calls_used)."""
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget or time.time() > deadline - 5:
                return None, None, 0

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(dim + 1))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) * (dim + 1) + 2)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) * (dim + mueff)))
            cs = (mueff + 2) / (dim + mueff + 5)

            sigma = sigma0
            xmean = np.array(x0, dtype=float)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            local_best_x = xmean.copy()
            local_best_f = np.inf
            calls_this_run = 0
            evals_remaining = max_evals

            # Evaluate initial point
            if n_calls < budget and time.time() < deadline:
                fx = safe_func(xmean)
                calls_this_run = 1
                local_best_f = fx
                local_best_x = xmean.copy()
                evals_remaining -= 1

            while evals_remaining > 0:
                if time.time() > deadline - 5 or n_calls >= budget:
                    break

                # Generate samples
                samples = []
                for i in range(lam):
                    if n_calls + len(samples) >= budget or time.time() > deadline - 3:
                        break
                    z = rng.standard_normal(dim)
                    xi = xmean + sigma * (B @ (D * z))
                    xi = np.clip(xi, -5, 5)
                    samples.append(xi)

                if len(samples) < 2:
                    break

                # Evaluate samples
                fvals = []
                for xi in samples:
                    if n_calls >= budget or time.time() > deadline - 3:
                        break
                    fi = safe_func(xi)
                    fvals.append(fi)
                    calls_this_run += 1
                    evals_remaining -= 1

                if len(fvals) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                x_sorted = [samples[i] for i in idx]
                f_sorted = [fvals[i] for i in idx]

                # Update local best
                if f_sorted[0] < local_best_f:
                    local_best_f = f_sorted[0]
                    local_best_x = x_sorted[0].copy()

                # Update mean
                if len(fvals) >= mu:
                    xnew = np.sum([w[i] * x_sorted[i] for i in range(min(mu, len(fvals)))], axis=0)
                else:
                    xnew = np.mean(x_sorted, axis=0)

                # Update evolution paths
                diff = (xnew - xmean) / (sigma + 1e-15)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D * diff))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_this_run // lam + 1) + 1e-15))) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1)
                if hsig:
                    pc = (1 - c1) * pc + c1 * diff
                else:
                    pc = (1 - c1) * pc

                # Update covariance
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (delta_hsig - c1 + c1 * c1) * np.eye(dim)) + \
                    cmu * np.sum([w[i] * np.outer(x_sorted[i] - xmean, x_sorted[i] - xmean) for i in range(min(mu, len(fvals)))], axis=0)

                # Periodic re-decomposition
                if calls_this_run % (10 * dim) == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-20, None)
                        C = (eigvecs * eigvals) @ eigvecs.T
                        B = eigvecs
                        D = np.sqrt(eigvals)
                    except np.linalg.LinAlgError:
                        pass

                xmean = xnew
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(dim + 1 - 1) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

            return local_best_x, local_best_f, calls_this_run

        # Phase 1: Multi-start exploration with short CMA-ES runs
        # Use diverse starting points to escape local minima of Katsuura
        n_restarts = 4
        budget_per_restart = budget // (n_restarts + 2)  # Reserve budget for final exploitation
        budget_per_restart = min(budget_per_restart, 80)

        starting_points = []
        # Diverse starting points
        starting_points.append(np.zeros(dim))
        starting_points.append(rng.uniform(-4, 4, dim))
        starting_points.append(rng.uniform(-5, 5, dim))
        starting_points.append(rng.uniform(-5, 5, dim))

        for i in range(n_restarts):
            if time.time() > deadline - 60 or n_calls >= budget - 100:
                break

            x0 = starting_points[i]
            sigma0 = 1.5  # Moderate initial step size for exploration

            bx, bf, _ = cma_es_run(x0, sigma0, budget_per_restart, seed_offset=i)

        # Phase 2: Exploit the best found with a longer CMA-ES run
        if best_x is not None and time.time() < deadline - 10 and n_calls < budget - 20:
            remaining = budget - n_calls
            sigma_final = 0.5  # Smaller step for exploitation
            cma_es_run(best_x, sigma_final, remaining, seed_offset=100)

        # Phase 3: Final fine-tuning if budget remains
        if best_x is not None and time.time() < deadline - 5 and n_calls < budget - 10:
            remaining = budget - n_calls
            if remaining > 20:
                sigma_tight = 0.1
                cma_es_run(best_x, sigma_tight, remaining, seed_offset=200)

        # Ensure at least one call was made
        if n_calls == 0:
            x0 = rng.uniform(-5, 5, dim)
            best_f = func(x0)
            best_x = x0
            n_calls = 1

        return best_x, best_f
