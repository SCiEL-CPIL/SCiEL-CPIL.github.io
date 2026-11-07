import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        lam = 2 * (dim + 1)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # Initialize
        x = np.zeros(dim)
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # First evaluation
        best_x = x.copy()
        best_f = func(x)
        total_calls = 1

        # Reserve some budget for local polish
        polish_budget = 50
        cma_budget = budget - polish_budget

        gen = 0
        while total_calls < cma_budget and time_ok():
            gen += 1
            # Generate lam candidates
            candidates = []
            for i in range(lam):
                if not time_ok() or total_calls >= cma_budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                fy = func(y)
                total_calls += 1
                candidates.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(candidates) < mu:
                break

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])
            # Take mu best
            y_best = np.array([c[1] for c in candidates[:mu]])
            # Weighted mean
            m = np.sum(weights[:, None] * y_best, axis=0)

            # Update evolution paths
            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / old_sigma * (B @ (D * (m - x)))

            # Check if ps is within bounds
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * total_calls / dim)) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (m - x) / old_sigma
                delta_h = 0
            else:
                pc = (1 - c_c) * pc
                delta_h = 1

            # Update covariance matrix
            C = B @ (D[:, None] * (D[None, :] * B.T))
            C_new = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2
            # Eigen decomposition
            D2, B2 = np.linalg.eigh(C_new)
            # Clip negative eigenvalues
            D2 = np.maximum(D2, 1e-10)
            D = np.sqrt(D2)
            B = B2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e6)

            # Update mean
            x = m

        # Local Polish Phase
        if total_calls < budget and time_ok():
            # Use Nelder-Mead for derivative-free local optimization
            # Wrap func to count calls
            def wrapped_func(x_polish):
                nonlocal total_calls
                if total_calls >= budget or not time_ok():
                    return np.inf
                val = func(x_polish)
                total_calls += 1
                return val

            # Initial simplex: best_x plus small perturbations
            # Nelder-Mead in scipy handles the initial simplex generation if not provided,
            # but we can provide it to be safe and centered on best_x.
            # Default initial simplex is based on initial point.
            
            try:
                # maxiter is a limit on iterations, not function calls.
                # We rely on total_calls check in wrapped_func.
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
