"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio-based CMA-ES with adaptive sigma, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = 5
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = [0]

        # Helper to track best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_deadline():
            return time.time() >= deadline or call_count[0] >= budget

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration with random samples to find a good starting point
        n_init = min(20, budget)
        for _ in range(n_init):
            if check_deadline():
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            call_count[0] += 1
            update_best(x, fx)

        # Phase 2: CMA-ES from the best found point
        if not check_deadline():
            self._cma_es(func, rng, x0=best_x.copy(), sigma0=1.5,
                         remaining_calls=budget - call_count[0],
                         deadline=deadline, update_best=update_best,
                         call_count=call_count)

        # Phase 3: If we have budget left, do local refinement with (1+1)-ES
        if not check_deadline():
            remaining = budget - call_count[0]
            if remaining > 10:
                self._es_local(func, rng, x0=best_x.copy(), sigma0=0.5,
                               remaining_calls=min(remaining, 100),
                               deadline=deadline, update_best=update_best,
                               call_count=call_count)

        # Phase 4: Final polish with very small steps
        if not check_deadline():
            remaining = budget - call_count[0]
            if remaining > 5:
                self._es_fine(func, rng, x0=best_x.copy(), sigma0=0.1,
                              remaining_calls=min(remaining, 50),
                              deadline=deadline, update_best=update_best,
                              call_count=call_count)

        return best_x, best_f

    def _cma_es(self, func, rng, x0, sigma0, remaining_calls, deadline, update_best, call_count):
        """Robust CMA-ES implementation."""
        dim = 5
        x = x0.copy()
        sigma = sigma0

        # CMA-ES parameters for dim=5
        mu = 3  # number of parents
        lam = 6  # number of offspring
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Precompute C^{1/2}
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        fx = func(x)
        update_best(x, fx)
        call_count[0] += 1

        gen = 0
        while call_count[0] < remaining_calls + 1:  # +1 because we already made 1 call
            if time.time() >= deadline:
                break

            # Generate offspring
            ys = []
            fs = []
            for i in range(lam):
                if time.time() >= deadline or call_count[0] >= remaining_calls + 1:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count[0] += 1
                ys.append(y)
                fs.append(fy)

            if len(fs) < mu:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx[:mu]]
            fs = fs[idx[:mu]]

            # Update best
            update_best(ys[0], fs[0])

            # Weighted mean
            y_w = np.sum(weights[:, None] * ys, axis=0)

            # Update evolution paths
            delta_x = (y_w - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_x

            # Update covariance matrix using standard formula
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_diff = (ys[i] - x) / sigma
                H += weights[i] * np.outer(y_diff, y_diff)

            C = (1 + c_c * (2 - c_c) * mu_eff - 1) * C + \
                c_sigma * (2 - c_sigma) * (np.outer(p_sigma, p_sigma) - C) + \
                (1 - (1 - c_c) * (2 - c_c) * mu_eff) * H

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update mean
            x = y_w

            gen += 1

            # Check convergence
            if sigma < 1e-7:
                break

        return

    def _es_local(self, func, rng, x0, sigma0, remaining_calls, deadline, update_best, call_count):
        """(1+1)-ES with moderate steps for local refinement."""
        dim = 5
        x = x0.copy()
        sigma = sigma0

        fx = func(x)
        update_best(x, fx)
        call_count[0] += 1

        while call_count[0] < remaining_calls + 1:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count[0] += 1

            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.1
            else:
                sigma *= 0.95

            sigma = np.clip(sigma, 0.01, 2.0)

        return

    def _es_fine(self, func, rng, x0, sigma0, remaining_calls, deadline, update_best, call_count):
        """(1+1)-ES with very small steps for final polish."""
        dim = 5
        x = x0.copy()
        sigma = sigma0

        fx = func(x)
        update_best(x, fx)
        call_count[0] += 1

        while call_count[0] < remaining_calls + 1:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count[0] += 1

            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.05
            else:
                sigma *= 0.95

            sigma = np.clip(sigma, 0.001, 0.5)

        return
# EVOLVE-BLOCK-END
