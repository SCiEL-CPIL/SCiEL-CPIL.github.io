"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and local refinement for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin

        best_x = None
        best_f = np.inf
        total_calls = 0
        n = self.dim

        def timed_func(x):
            nonlocal total_calls
            if total_calls >= self.budget:
                return np.inf
            fx = func(x)
            total_calls += 1
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_cma, func_wrapper):
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))
            # For dim=5, lam = max(4, int(4+3*1.609)) = max(4, 8) = 8
            # But with budget constraints, we might need to adjust
            if lam > budget_cma // 5:
                lam = max(2, budget_cma // 5)

            # CMA-ES parameters
            cmu = 1 / (n + 4 + 2 / lam)
            cphi = (4 + cmu) / (n + 4)
            c1 = (2 - cmu + 2 * min(1, (lam / 2 - 1) / (n + 2))) / ((n + 2) ** 2 + lam)
            cmu = 1 / (n + 4 + 2 / lam)
            cs = (4 + cmu) / (n + 4)
            ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1) - 1)) + cs

            # Initialize
            xmean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            calls_used = 0
            best_local_x = xmean.copy()
            best_local_f = func_wrapper(xmean)
            calls_used += 1

            if calls_used >= budget_cma:
                return best_local_x, best_local_f, calls_used

            # Main loop
            while calls_used < budget_cma:
                # Generate lambda samples
                samples = []
                for _ in range(lam):
                    z = np.random.standard_normal(n)
                    y = xmean + sigma * (B @ (D * z))
                    samples.append(y)

                # Evaluate samples
                fvals = []
                for y in samples:
                    fy = func_wrapper(y)
                    calls_used += 1
                    fvals.append(fy)
                    if calls_used >= budget_cma:
                        break

                if len(fvals) < lam:
                    break

                fvals = np.array(fvals)
                # Sort by fitness
                order = np.argsort(fvals)
                sorted_samples = [samples[i] for i in order]
                sorted_fvals = fvals[order]

                # Update best
                if sorted_fvals[0] < best_local_f:
                    best_local_f = sorted_fvals[0]
                    best_local_x = sorted_samples[0].copy()

                # Update mean
                mu = lam // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)

                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += w[i] * sorted_samples[i]
                xmean = new_mean

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * ((new_mean - xmean) / sigma)))
                # Actually, the update is: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * (B^{-1} * (new_mean - old_mean) / sigma)
                # Let me redo this properly
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (D ** -1) @ (B.T @ ((new_mean - xmean) / sigma))

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used + 1) / lam))) < (1.4 + 2 / (n + 1)) * chi_n
                hc = (1 - hsig) * (1 - cs)

                # Update covariance matrix
                C = (B @ np.diag(D ** 2) @ B.T)
                delta_hsig = (1 - hc) * cs * (2 - cs)
                rank_one = np.outer(new_mean - xmean, new_mean - xmean)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (B.T @ ((sorted_samples[i] - xmean) / sigma))
                    rank_mu += w[i] * np.outer(zi, zi)

                C = (1 + c1 * delta_hsig - c1) * C + c1 * rank_one + cmu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-15)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    break

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # Convergence check
                if np.linalg.norm(new_mean - xmean) < 1e-12 and sigma < 1e-8:
                    break

            return best_local_x, best_local_f, calls_used

        # Local refinement using Nelder-Mead
        def nelder_mead(x0, budget_lm, func_wrapper):
            n = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            f0 = func_wrapper(x0)
            for i in range(n):
                xi = x0.copy()
                step = 0.05 if abs(x0[i]) < 1e-10 else 0.05 * abs(x0[i])
                xi[i] += step
                simplex.append(xi)

            fvals = [f0]
            for i in range(1, n + 1):
                fvals.append(func_wrapper(simplex[i]))

            best_idx = np.argmin(fvals)
            best_x = simplex[best_idx].copy()
            best_f = fvals[best_idx]

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            iterations = 0
            max_iter = budget_lm

            while iterations < max_iter:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                best_idx = 0
                worst_idx = n
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[worst_idx])
                fr = func_wrapper(xr)

                if fvals[best_idx] <= fr < fvals[worst_idx - 1]:
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
                elif fr < fvals[best_idx]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    fe = func_wrapper(xe)
                    if fe < fr:
                        simplex[worst_idx] = xe
                        fvals[worst_idx] = fe
                    else:
                        simplex[worst_idx] = xr
                        fvals[worst_idx] = fr
                else:
                    # Contraction
                    if fr < fvals[worst_idx]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[worst_idx] - centroid)
                    fc = func_wrapper(xc)
                    if fc < min(fr, fvals[worst_idx]):
                        simplex[worst_idx] = xc
                        fvals[worst_idx] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = func_wrapper(simplex[i])

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                # Convergence check
                if np.std(fvals) < 1e-12:
                    break

                iterations += 1

            return best_x, best_f

        # Multi-start CMA-ES
        rng = np.random.default_rng(0)
        n_starts = 3
        budget_per_start = self.budget // n_starts
        # Reserve some budget for local refinement
        budget_cma = int(budget_per_start * 0.7)
        budget_lm = budget_per_start - budget_cma

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            if total_calls >= self.budget:
                break

            # Initialize starting point
            if start_idx == 0:
                x0 = np.zeros(n)
            elif start_idx == 1:
                x0 = rng.uniform(-3, 3, n)
            else:
                x0 = rng.uniform(-4, 4, n)

            # Run CMA-ES
            x_cma, f_cma, calls_cma = cma_es(x0, 2.0, budget_cma, timed_func)

            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma.copy()

            # Run local refinement if we have budget left
            if total_calls < self.budget and budget_lm > 0:
                x_lm, f_lm = nelder_mead(x_cma, budget_lm, timed_func)
                if f_lm < best_f:
                    best_f = f_lm
                    best_x = x_lm.copy()

        # Final check: if we still have budget, do more local refinement
        remaining = self.budget - total_calls
        if remaining > 0 and best_x is not None and time.time() < deadline:
            x_final, f_final = nelder_mead(best_x, remaining, timed_func)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
