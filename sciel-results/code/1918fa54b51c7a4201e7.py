"""Optimizer for BBOB f18 (Schaffers F7 ill-conditioned) using CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using CMA-ES with adaptive restarts and Nelder-Mead refinement.
    Designed for ill-conditioned functions in 5D.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        rng = np.random.default_rng(42)
        evals = 0
        best_x = None
        best_f = float('inf')

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(center, sigma, max_evals, seed):
            """Run CMA-ES from given center and sigma."""
            nonlocal evals, best_x, best_f

            rng_local = np.random.default_rng(seed)
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

            m = center.copy()
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            # Initial evaluation of center
            if evals < budget and time.time() < deadline:
                f_center = func(m)
                evals += 1
                record(m, f_center)

            stagnation_counter = 0
            best_in_run = best_f
            no_improve_evals = 0

            while evals < budget and time.time() < deadline:
                # Compute C^(-1/2)
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12)
                    C_inv_sqrt = (B / np.sqrt(D)) @ B.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                # Sample offspring
                Y = np.zeros((lam, n))
                F = np.empty(lam)
                for i in range(lam):
                    if evals >= budget or time.time() > deadline:
                        break
                    z = rng_local.standard_normal(n)
                    Y[i] = m + sigma * (C_inv_sqrt @ z)
                    Y[i] = np.clip(Y[i], -5.0, 5.0)
                    F[i] = func(Y[i])
                    evals += 1
                    record(Y[i], F[i])

                if evals >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(F)
                Y_best = Y[order[:mu]]
                F_best = F[order[:mu]]

                # Check for improvement
                if F_best[0] < best_in_run - 1e-10:
                    best_in_run = F_best[0]
                    no_improve_evals = 0
                    stagnation_counter = 0
                else:
                    no_improve_evals += lam
                    stagnation_counter += 1

                # Detect stagnation
                if no_improve_evals > lam * 15 or stagnation_counter > 30:
                    break

                # Update mean
                m_old = m.copy()
                m = w @ Y_best.T
                m = np.clip(m, -5.0, 5.0)

                # Update path statistics
                m_diff = m - m_old
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ m_diff)

                hc = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * max(1, evals // lam))) <
                      (1.4 + c_c / n) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ m_diff)

                # Update covariance matrix
                rank_one = pc @ pc.T + (1 - hc) * c_sigma * (2 - c_sigma) * C
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (Y_best[i] - m_old)
                    rank_mu += w[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 0.001, 10.0)

            return best_x, best_f

        def nelder_mead_refine(x0, max_evals):
            """Refine using Nelder-Mead."""
            nonlocal evals, best_x, best_f

            if evals >= budget or time.time() > deadline:
                return

            # Create initial simplex
            simplex = [x0.copy()]
            f0 = func(x0)
            evals += 1
            record(x0, f0)

            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.25
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                if evals < budget and time.time() < deadline:
                    fi = func(xi)
                    evals += 1
                    record(xi, fi)

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            for _ in range(200):
                if evals >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                simple_f = [func(s) for s in simplex]
                # This is inefficient; let's track properly
                # Actually, let's use scipy's minimize instead for reliability

                break

            # Use scipy minimize for Nelder-Mead
            if evals < budget and time.time() < deadline:
                remaining = budget - evals
                if remaining > 10:
                    try:
                        result = minimize(
                            func, x0, method='Nelder-Mead',
                            options={'maxiter': min(100, remaining // 2),
                                     'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if result.fun < best_f:
                            best_x = result.x.copy()
                            best_f = result.fun
                            record(best_x, best_f)
                    except Exception:
                        pass

        # Phase 1: Initial exploration with multiple starting points
        n_starts = min(5, max(2, budget // 100))
        starts = []
        for i in range(n_starts):
            if evals >= budget or time.time() > deadline:
                break
            if i == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng.uniform(-4, 4, n)
            starts.append(x0)

        # Run CMA-ES from each start point
        for i, x0 in enumerate(starts):
            if evals >= budget or time.time() > deadline:
                break

            # Allocate remaining budget
            remaining = budget - evals
            if remaining < 20:
                break

            # Use a portion of remaining budget
            alloc = max(30, remaining // (n_starts - i))
            seed = 42 + i * 1000
            cma_es(x0, 1.5, alloc, seed)

        # Phase 2: Final refinement with Nelder-Mead
        if best_x is not None and evals < budget and time.time() < deadline:
            nelder_mead_refine(best_x, budget - evals)

        # Phase 3: Local coordinate descent refinement
        if evals < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.1

            for _ in range(10):  # 10 passes
                if evals >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(n):
                    if evals >= budget or time.time() > deadline:
                        break
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[d] += step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    if evals < budget and time.time() < deadline:
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < fx:
                            x, fx = x_trial, f_trial
                            record(x, fx)
                            improved = True
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] -= step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    if evals < budget and time.time() < deadline:
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < fx:
                            x, fx = x_trial, f_trial
                            record(x, fx)
                            improved = True

                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break
                else:
                    step *= 1.2

        return best_x, best_f
# EVOLVE-BLOCK-END
