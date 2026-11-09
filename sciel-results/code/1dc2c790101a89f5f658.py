"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with LHS initialization and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling in [-5, 5]^dim."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0

    def _cmaes(self, func, x0, sigma0, budget, dim, rng, deadline):
        """CMA-ES implementation."""
        n = dim
        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(n))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((n + 2) ** 2 + mu_eff))
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # Initial step size
        sigma = sigma0
        x_best = x0.copy()
        f_best = func(x_best)
        
        # Precompute for Cholesky
        def _cholesky_update(C, C_inv, c_1, c_mu, weights, y_w, p_sigma, p_c, n, mu, sigma):
            """Update C and C_inv using rank-one and rank-mu updates."""
            # Rank-one update
            C += c_1 * (p_c @ p_c.T - (1 - c_c) * C)
            # Rank-mu update
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_w[i], y_w[i])
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Invert
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(n)
            # Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(n)
                C_sqrt_inv = np.eye(n)
            return C, C_inv, C_sqrt, C_sqrt_inv

        count = 1
        for gen in range(budget // lambda_ + 1):
            if count >= budget or time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                x_new = x_best + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                count += 1
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
            
            if len(xs) < lambda_:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = np.array([xs[i] for i in order])
            fs_sorted = [fs[i] for i in order]
            ys_sorted = np.array([ys[i] for i in order])
            
            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                x_best = xs_sorted[0].copy()
            
            # Weighted mean
            y_w = np.zeros((mu, n))
            for i in range(mu):
                y_w[i] = (xs_sorted[i] - x_best) / sigma if sigma > 0 else np.zeros(n)
            x_mean = x_best + sigma * (weights @ y_w)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_w.T @ weights).T
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ y_w.T @ weights).T
            
            # Step size control
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count + 1) / lambda_)) - 1))
            
            # Update covariance
            C, C_inv, C_sqrt, C_sqrt_inv = _cholesky_update(
                C, C_inv, c_1, c_mu, weights, y_w, p_sigma, p_c, n, mu, sigma
            )
            
            # Move center
            x_best = x_mean
        
        return x_best, f_best, count

    def _nelder_mead(self, func, x0, budget, dim, deadline):
        """Nelder-Mead simplex method for final polish."""
        n = dim
        # Initial simplex
        simplex = [x0.copy()]
        f0 = func(x0)
        count = 1
        simplex[0] = x0.copy()
        
        for i in range(n):
            x_i = x0.copy()
            x_i[i] += 0.5 if x0[i] < 4.5 else -0.5
            x_i = np.clip(x_i, -5, 5)
            simplex.append(x_i)
            count += 1
        
        fs = [func(x) for x in simplex]
        count += n
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        
        for _ in range(budget - count):
            if time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            count += 1
            
            if f_r < fs[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                count += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fs[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fs[-1] = f_r
            elif f_r < fs[-2]:
                simplex[-1] = x_r
                fs[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                count += 1
                if f_c < fs[-1]:
                    simplex[-1] = x_c
                    fs[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fs[i] = func(simplex[i])
                        count += 1
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
        
        # Return best
        best_idx = np.argmin(fs)
        return simplex[best_idx], fs[best_idx], count

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: LHS spread for early progress
        n_lhs = min(50, budget // 10)
        lhs_points = self._lhs_sample(n_lhs, dim, rng)
        lhs_vals = [func(p) for p in lhs_points]
        best_idx = np.argmin(lhs_vals)
        x_best = lhs_points[best_idx].copy()
        f_best = lhs_vals[best_idx]
        count = n_lhs
        
        # Phase 2: CMA-ES from best LHS point
        if count < budget - 10:
            cma_budget = budget - count - 10
            x_cma, f_cma, cma_count = self._cmaes(
                func, x_best, 1.0, cma_budget, dim, rng, deadline
            )
            if f_cma < f_best:
                x_best = x_cma
                f_best = f_cma
            count += cma_count
        
        # Phase 3: Nelder-Mead polish
        if count < budget - 5 and time.time() < deadline:
            nm_budget = min(budget - count, 50)
            x_nm, f_nm, nm_count = self._nelder_mead(
                func, x_best, nm_budget, dim, deadline
            )
            if f_nm < f_best:
                x_best = x_nm
                f_best = f_nm
            count += nm_count
        
        return x_best, f_best
# EVOLVE-BLOCK-END
