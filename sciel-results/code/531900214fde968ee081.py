"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f7 (Step Ellipsoidal) with efficient Cholesky updates."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        sigma = 2.0
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        mean = np.zeros(dim)
        calls = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1

        if calls >= self.budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Cholesky factor of C (initially I)
        L = np.eye(dim)

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        while calls < self.budget and time.time() < deadline:
            # Generate population using Cholesky factor
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)

            # Evaluate population
            fvals = []
            for x in pop:
                fx = _eval(x)
                if fx is None:
                    fvals.append(np.inf)
                else:
                    fvals.append(fx)

            if calls >= self.budget or time.time() > deadline:
                break

            # Sort by fitness
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            sorted_fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

            # Compute C^-1/2 using Cholesky: C = L L^T, so C^-1/2 = (L^T)^{-1} L^{-1}
            # More efficiently: solve L y = b, then solve L^T x = y
            # For a vector v: C^{-1/2} v = (L^T)^{-1} (L^{-1} v)
            
            # Compute inv_sqrt_C @ diff where diff = (new_mean - mean) / sigma
            diff = (new_mean - mean) / sigma
            
            # Solve L y = diff
            y_temp = np.linalg.solve(L, diff)
            # Solve L^T x = y_temp
            inv_sqrt_C_diff = np.linalg.solve(L.T, y_temp)

            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * inv_sqrt_C_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C_diff

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            # Recompute Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small jitter if not positive definite
                C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)

            mean = new_mean

        # Final local refinement using Nelder-Mead if budget allows
        if calls < self.budget and time.time() < deadline - 10:
            remaining = self.budget - calls
            if remaining > 20:
                def wrapped_func(x):
                    nonlocal calls, best_f, best_x
                    if calls >= self.budget:
                        return best_f
                    if time.time() > deadline:
                        return best_f
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                try:
                    from scipy.optimize import minimize
                    res = minimize(wrapped_func, best_x.copy(), method='Nelder-Mead',
                                   options={'maxiter': min(remaining, 100), 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
