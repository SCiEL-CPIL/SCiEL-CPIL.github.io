"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Single continuous CMA-ES with adaptive step size, no restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_f = np.inf
        best_x = None

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return (time.time() - start_time) > deadline

        # --- Single CMA-ES run ---
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters for dim=5
        pop_size = 4 * int(dim / 2 + 1)
        pop_size = max(4, min(pop_size, 20))
        
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cs = (dim + 1.7) / (dim + 22)
        c1 = 2 / ((dim + 1.3) ** 2 + 10)
        cmu = min(0.3 - c1, 2 / ((dim + 2) ** 2 - 1))
        dsigma = 1 + 2 * max(0, np.sqrt(dim) - 1) / (dim + 3)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize at origin
        x_mean = np.zeros(dim)
        f_mean = call_func(x_mean)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        it = 0
        max_iter = 200  # safety cap on iterations

        while calls_made < budget and not check_deadline() and it < max_iter:
            # Generate offspring
            y_list = []
            f_y_list = []
            for k in range(pop_size):
                if calls_made >= budget or check_deadline():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                y_list.append(y)
                f_y_list.append(fy)

            if len(y_list) < 2:
                break

            # Sort by fitness
            indices = np.argsort(f_y_list)
            y_sorted = np.array([y_list[i] for i in indices])
            f_y_sorted = [f_y_list[i] for i in indices]

            # Weights
            w = np.log(pop_size / 2 + 1) - np.arange(1, pop_size + 1)
            w = w / w.sum()
            mu_eff = 1 / np.sum(w[:int(2 * pop_size / 2)] ** 2)

            # Update mean
            y_mean_new = np.sum(w[:int(2 * pop_size / 2)][:, None] * y_sorted[:int(2 * pop_size / 2)], axis=0)

            # Update path cumulation
            pc = (1 - cc) * pc + cc * (1 / sigma) * (y_mean_new - x_mean)
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (it + 1)))) / chi_n < (1.4 + 2 / (dim + 1))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * hsig) * (1 / sigma) * (y_mean_new - x_mean)

            # Update covariance matrix
            delta = (1 + hsig * cs * (2 - cs)) * c1 * (np.outer(pc, pc) - C)
            rank_mu = (y_sorted[:int(2 * pop_size / 2)] - x_mean)
            delta += cmu * np.sum(w[:int(2 * pop_size / 2)] * np.outer(rank_mu, rank_mu), axis=0)
            C = (1 + c1 + cmu - c1) * C + delta

            # Ensure C is positive definite and symmetric
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += (1e-12 - eigvals.min()) * np.eye(dim)

            # Update sigma
            sigma *= np.exp(1 / dsigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Update mean
            x_mean = y_mean_new
            f_mean = f_y_sorted[0]

            # Recompute Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            it += 1

            # Stop if sigma is too small (converged) or too large
            if sigma < 1e-10:
                break

        # --- Final local refinement with Nelder-Mead if budget remains ---
        if best_x is not None and calls_made < budget and not check_deadline():
            from scipy.optimize import minimize as sp_minimize
            x0 = best_x.copy()
            remaining = budget - calls_made
            max_iter_nm = max(50, remaining // (dim + 1))

            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                calls_made += 1
                return float(func(x))

            try:
                res = sp_minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': max_iter_nm, 'xatol': 1e-14, 'fatol': 1e-14, 'maxfev': remaining})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
