"""CMA-ES with adaptive restarts and BFGS polish for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    CMA-ES based optimizer with adaptive restarts and BFGS local polish
    for BBOB f19 (Griewank-Rosenbrock) in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_iter, calls_made, best_f, best_x, budget, deadline):
        """
        Run CMA-ES from x0 with initial sigma sigma0.
        Returns (calls_made, best_f, best_x).
        """
        n = self.dim
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)

        # CMA-ES parameters
        lam = 4 * (n + 2)  # population size
        mu = lam // 2      # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4.0 - mueff / n + 1.0) / (n + 4.0 + mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 - mueff / (2.0 * n) + 1.0) / ((n + 1.0) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0) - 1.0)) + cs

        # State variables
        mean = x0.copy().astype(np.float64)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        sqrt_eig = np.sqrt(np.linalg.eigvalsh(C))
        dcor = 1.0 + 2.0 * max(0.0, (mueff - 1.0) / (n + 1.0) - 1.0) + cs

        # Initial population
        best_local_f = best_f
        best_local_x = best_x.copy() if best_x is not None else mean.copy()
        local_calls = calls_made

        def eval_point(x):
            nonlocal local_calls, best_local_f, best_local_x
            if local_calls >= budget or time.time() > deadline - 2:
                return best_local_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            local_calls += 1
            if val < best_local_f:
                best_local_f = val
                best_local_x = x.copy()
            return val

        for it in range(max_iter):
            if local_calls >= budget or time.time() > deadline - 3:
                break

            # Sample population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Symmetrize and retry
                C = 0.5 * (C + C.T)
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

            z = rng.standard_normal((lam, n))
            pop = mean + sigma * (z @ L.T)

            # Evaluate population
            fvals = np.array([eval_point(p) for p in pop])

            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]
            fvals_sorted = fvals[idx]

            # Update mean (weighted recombination)
            mean_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)

            # Update step size control variable
            # Compute C^{-1/2} pc
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.linalg.pinv(L)
            pc = (1.0 - cs) * pc + cs * damps * L_inv.T @ ((mean - mean_old) / (sigma * np.linalg.norm(L_inv.T @ np.eye(n))))
            # Simpler: pc = (1-cs)*pc + cs*damps * (mean - mean_old)/sigma
            # Actually the standard formula uses C^{-1/2}(mean - mean_old)/sigma
            # Let me use the simpler approximation
            pc = (1.0 - cs) * pc + cs * damps * (mean - mean_old) / max(sigma, 1e-20)

            # Update evolution path for covariance
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} (mean - mean_old)/sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * L_inv.T @ ((mean - mean_old) / max(sigma, 1e-20))

            # Rank-mu update
            v = (pop_sorted[:mu] - mean_old) / max(sigma, 1e-20)
            C_new = C + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * v[:, np.newaxis, :] @ v[:, np.newaxis, :].T, axis=0)
            C = 0.5 * (C_new + C_new.T)

            # Update step size (Hessian-free CSA)
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            sigma *= np.exp((cs / damps) * (norm_ps / chi_n - 1.0))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Check convergence
            eigvals = np.linalg.eigvalsh(C)
            if np.max(eigvals) / max(np.min(eigvals), 1e-20) > 1e14:
                break
            if sigma < 1e-12:
                break

        return local_calls, best_local_f, best_local_x

    def _bfgs_polish(self, func, x0, calls_made, best_f, best_x, budget, deadline):
        """Run BFGS polish from x0. Returns (calls_made, best_f, best_x)."""
        n = self.dim
        local_best_f = best_f
        local_best_x = best_x.copy()
        local_calls = calls_made

        def bfgs_func(x):
            nonlocal local_calls, local_best_f, local_best_x
            if local_calls >= budget or time.time() > deadline - 1:
                return local_best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            local_calls += 1
            if val < local_best_f:
                local_best_f = val
                local_best_x = x.copy()
            return val

        try:
            res = minimize(bfgs_func, x0, method='BFGS',
                           options={'maxiter': 50, 'gtol': 1e-8})
            if res.fun < local_best_f:
                local_best_f = res.fun
                local_best_x = np.clip(res.x, -5.0, 5.0).copy()
        except Exception:
            pass

        return local_calls, local_best_f, local_best_x

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Initial evaluation at origin
        val = safe_call(np.zeros(n))
        if val is not None:
            best_f = val
            best_x = np.zeros(n).copy()

        # Multi-restart CMA-ES
        # First run: larger sigma to explore
        sigma0 = 2.0
        max_iter = 80
        calls_made, best_f, best_x = self._cma_es(
            func, np.zeros(n), sigma0, max_iter, calls_made, best_f, best_x, budget, deadline
        )

        # Second run: smaller sigma, centered on best found
        if calls_made < budget - 20 and time.time() < deadline - 10:
            sigma1 = 0.8
            max_iter2 = 60
            calls_made, bf2, bx2 = self._cma_es(
                func, best_x.copy(), sigma1, max_iter2, calls_made, best_f, best_x, budget, deadline
            )
            if bf2 < best_f:
                best_f = bf2
                best_x = bx2

        # Third run: even smaller sigma for fine-tuning
        if calls_made < budget - 15 and time.time() < deadline - 8:
            sigma2 = 0.3
            max_iter3 = 40
            calls_made, bf3, bx3 = self._cma_es(
                func, best_x.copy(), sigma2, max_iter3, calls_made, best_f, best_x, budget, deadline
            )
            if bf3 < best_f:
                best_f = bf3
                best_x = bx3

        # BFGS polish
        if calls_made < budget - 10 and time.time() < deadline - 5 and best_x is not None:
            calls_made, best_f, best_x = self._bfgs_polish(
                func, best_x.copy(), calls_made, best_f, best_x, budget, deadline
            )

        # Final (1+1)-ES with tiny steps
        if calls_made < budget and time.time() < deadline - 1 and best_x is not None:
            rng = np.random.default_rng(12345)
            polish_x = best_x.copy()
            polish_f = best_f
            polish_sigma = 0.01

            while calls_made < budget and time.time() < deadline - 0.5:
                step = polish_sigma * rng.standard_normal(n)
                candidate = np.clip(polish_x + step, -5, 5)
                candidate_f = func(candidate)
                calls_made += 1

                if candidate_f < polish_f:
                    polish_x = candidate
                    polish_f = candidate_f
                    polish_sigma *= 1.1
                    if polish_f < best_f:
                        best_f = polish_f
                        best_x = polish_x.copy()
                else:
                    polish_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
