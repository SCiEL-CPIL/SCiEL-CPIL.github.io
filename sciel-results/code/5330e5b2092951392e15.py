"""Combined optimizer: CMA-ES global search with Nelder-Mead local polish for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start and Nelder-Mead local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 285  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def nelder_mead_local(start_x, max_evals):
            """Quick local NM polish around start_x."""
            nonlocal best_x, best_f, deadline, self
            if self._call_count >= self.budget or time.time() > deadline:
                return
            x0 = np.clip(np.asarray(start_x, dtype=float), -5, 5)
            f0 = counted_func(x0)
            if f0 == np.inf:
                return

            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                step = 0.3 if x0[i] < 4.7 else -0.3
                xi[i] = x0[i] + step
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, self.dim + 1):
                simplex_f.append(counted_func(simplex[i]))

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            iters = 0
            while iters < max_evals and self._call_count < self.budget and time.time() < deadline:
                iters += 1
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    simplex[-1] = xe if fe < fr else xr
                    simplex_f[-1] = fe if fe < fr else fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])

        def cma_es(x0, max_calls, initial_sigma=1.0):
            """CMA-ES implementation with adaptive restart."""
            nonlocal best_x, best_f, deadline, self
            n = self.dim
            if max_calls < 10 or time.time() > deadline:
                return None, np.inf

            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            lambda_ = min(lambda_, max(2, max_calls - 1))
            if lambda_ < 2:
                lambda_ = 2

            # Strategy parameters
            mu = max(1, lambda_ // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initial sigma
            sigma = initial_sigma

            # Mean and covariance matrix
            m = np.clip(np.asarray(x0, dtype=float), -5, 5)
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt_inv = np.eye(n)

            def update_chol(C_mat):
                try:
                    L = np.linalg.cholesky(C_mat)
                    C_inv_mat = np.linalg.inv(L @ L.T)
                    C_sqrt_inv_mat = np.linalg.inv(L)
                    return C_inv_mat, C_sqrt_inv_mat
                except np.linalg.LinAlgError:
                    C_reg = C_mat + 1e-6 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(C_reg)
                        C_inv_mat = np.linalg.inv(L @ L.T)
                        C_sqrt_inv_mat = np.linalg.inv(L)
                        return C_inv_mat, C_sqrt_inv_mat
                    except np.linalg.LinAlgError:
                        return None, None

            C_inv, C_sqrt_inv = update_chol(C)
            if C_inv is None:
                return None, np.inf

            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)

            local_best_x = m.copy()
            local_best_f = np.inf

            calls_used = 0
            rng_local = np.random.default_rng(12345 + int(time.time() * 1000) % 100000)

            while calls_used < max_calls and self._call_count < self.budget and time.time() < deadline - 5:
                # Generate and evaluate population
                pops = []
                fvals = []
                for i in range(lambda_):
                    if calls_used >= max_calls or self._call_count >= self.budget or time.time() > deadline - 5:
                        break
                    z = rng_local.standard_normal(n)
                    x = m + sigma * (C_sqrt_inv @ z)
                    x = np.clip(x, -5, 5)
                    fx = counted_func(x)
                    if fx == np.inf:
                        break
                    pops.append(x)
                    fvals.append(fx)
                    calls_used += 1
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()

                if len(fvals) < lambda_:
                    break

                # Sort by function value
                indices = np.argsort(fvals)
                pops = [pops[i] for i in indices]
                fvals = [fvals[i] for i in indices]

                # Update best
                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = pops[0].copy()

                # Update mean
                m_old = m.copy()
                m = np.sum([weights[i] * pops[i] for i in range(mu)], axis=0)
                m = np.clip(m, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv @ ((m - m_old) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** ((calls_used + 1) / (cs * n)))) < \
                       (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (m - m_old) / sigma

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + cmu * np.sum([
                    weights[i] * np.outer(pops[i] - m_old, pops[i] - m_old)
                    for i in range(mu)
                ], axis=0) + (c1 + c1 * delta_hsig - cmu) * np.outer(pc, pc)

                # Ensure positive definiteness
                C = (C + C.T) / 2
                C_inv, C_sqrt_inv = update_chol(C)
                if C_inv is None:
                    break

                # Update sigma
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt((n + 1) / (n - 1)) - 1))

                # Adaptive restart: if sigma is too small or no progress
                if sigma < 1e-8:
                    break

                # Check convergence
                if len(fvals) >= mu and np.std(fvals[:mu]) < 1e-10:
                    break

            return local_best_x, local_best_f

        # Main strategy: Multi-start CMA-ES with NM polish
        rng = np.random.default_rng(42)

        # Phase 1: Generate diverse starting points
        n_starts = 5
        starts = []
        starts.append(np.zeros(self.dim))  # Center
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all starts
        start_results = []
        for s in starts:
            if self._call_count >= self.budget or time.time() > deadline:
                break
            fs = counted_func(s)
            if fs != np.inf:
                start_results.append((fs, s.copy()))
                if fs < best_f:
                    best_f = fs
                    best_x = s.copy()

        if not start_results:
            return best_x, best_f

        start_results.sort(key=lambda x: x[0])

        # Phase 2: CMA-ES refinement from best starts
        remaining = self.budget - self._call_count
        if remaining > 20:
            # Use CMA-ES from the best 2 starts
            n_cma = min(2, len(start_results))
            budget_per_cma = max(30, remaining // (n_cma + 1))

            for i in range(n_cma):
                if self._call_count >= self.budget or time.time() > deadline - 10:
                    break

                fs, xs = start_results[i]
                x_opt, f_opt = cma_es(xs, budget_per_cma, initial_sigma=1.0)

                if x_opt is not None and f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        # Phase 3: Final local refinement around best_x if budget remains
        remaining = self.budget - self._call_count
        if remaining > 15 and best_x is not None and time.time() < deadline - 5:
            x_opt, f_opt = cma_es(best_x, remaining, initial_sigma=0.1)
            if x_opt is not None and f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 4: NM polish from best point
        if self._call_count < self.budget and best_x is not None and time.time() < deadline - 3:
            polish_budget = self.budget - self._call_count
            if polish_budget > 10:
                nelder_mead_local(best_x, polish_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
