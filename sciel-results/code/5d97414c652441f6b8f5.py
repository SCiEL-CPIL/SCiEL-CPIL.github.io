"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and Nelder-Mead polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                f = func(np.asarray(x, dtype=np.float64))
                f = float(f)
                if np.isfinite(f):
                    if f < best_f:
                        best_f = f
                        best_x = np.array(x, dtype=np.float64)
                    return f
                return np.inf
            except Exception:
                return np.inf

        def cma_es_run(x0, sigma0, remaining_budget, rng):
            """Run CMA-ES with given initial point, sigma, and budget."""
            nonlocal call_count, best_x, best_f
            n = dim
            pop_size = int(4 + 3 * np.log(n))
            pop_size = min(pop_size, max(2, remaining_budget // 2))
            if pop_size < 2:
                pop_size = 2

            # CMA-ES parameters
            cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
            cs = (n + 1.3) / (n + n + 7)
            c1 = 2 / ((n + 1.3) ** 2 + n)
            cmu = min(0.2 - c1, 2 / ((n + 2) ** 2 - n))
            damps = 1 + 2 * max(0, np.sqrt((cs + 1) * (n - 1) / (cs + 1)) - 1) + cs
            ccov1 = c1
            ccovmu = cmu

            # Initialize
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            normC = np.sqrt(np.trace(C) / n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Evaluate initial population
            xs = [x]
            fs = [safe_call(x)]

            for gen in range(remaining_budget // pop_size):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Generate population
                for i in range(pop_size):
                    if call_count >= budget:
                        break
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    xs.append(y)
                    fs.append(fy)

                if len(xs) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                xs = [xs[i] for i in idx]
                fs = [fs[i] for i in idx]

                # Update parameters
                mu_eff = sum(1.0 for i in range(pop_size // 2 + 1) if i < pop_size // 2 + 1)
                weights = np.zeros(pop_size // 2 + 1)
                for i in range(pop_size // 2 + 1):
                    weights[i] = np.log(pop_size / 2 + 1) - np.log(i + 1)
                mu_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)
                weights = weights / np.sum(weights)

                # Update mean
                x_old = x.copy()
                x = np.zeros(n)
                for i in range(pop_size // 2 + 1):
                    x += weights[i] * xs[i]
                x = np.clip(x, -5, 5)

                # Update step-size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / normC * (B_inv @ (x_old - x))
                # Approximate B_inv @ (x_old - x) using C^{-1/2}
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / normC * (C_inv_sqrt.T @ (x_old - x))
                except:
                    ps = (1 - cs) * ps

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < \
                       (1.4 + 2 / (n + 1)) * chi_n

                # Update covariance
                if hsig:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (x_old - x) / sigma
                else:
                    pc = (1 - cc) * pc

                delta = 1 + (2 * (hsig - 0.5) - 1) * ccovmu
                C = (1 + ccov1 * delta - ccov1 - ccovmu) * C
                for i in range(pop_size // 2 + 1):
                    diff = (xs[i] - x_old) / sigma
                    C += ccovmu * weights[i] * np.outer(diff, diff)
                if not hsig:
                    C += ccovmu * (1 - ccovmu) / ccov1 * np.outer(pc, pc) / (sigma ** 2) * (sigma ** 2)

                # Ensure positive definite
                try:
                    np.linalg.cholesky(C)
                except:
                    C = C + 1e-6 * np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

            return best_x, best_f

        def nelder_mead(x0, remaining_budget, rng):
            """Nelder-Mead local optimization."""
            nonlocal call_count, best_x, best_f
            n = dim
            if remaining_budget < 10:
                return

            # Initialize simplex
            simplex = [np.array(x0, dtype=np.float64)]
            f0 = safe_call(simplex[0])
            for i in range(n):
                xi = simplex[0].copy()
                xi[i] += 0.1 if xi[i] < 0 else -0.1
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                safe_call(xi)

            fs = [safe_call(s) for s in simplex]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5

            for _ in range(remaining_budget - 2 * n - 1):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Sort
                idx = np.argsort(fs)
                simplex = [simplex[i] for i in idx]
                fs = [fs[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
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
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + rho * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs[i] = safe_call(simplex[i])

            return best_x, best_f

        # Main optimization loop
        rng = np.random.default_rng(0)

        # Phase 1: CMA-ES with multiple restarts
        n_restarts = 3
        per_restart_budget = budget // (n_restarts + 1)
        polish_budget = budget - n_restarts * per_restart_budget

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Initialize: first restart at origin, others random
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.5

            cma_es_run(x0, sigma0, per_restart_budget, rng)

        # Phase 2: Nelder-Mead polish on best point
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                nelder_mead(best_x, remaining, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
