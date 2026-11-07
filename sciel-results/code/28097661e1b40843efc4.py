import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def safe_eval(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget or not time_ok():
                return np.inf
            val = func(x)
            total_calls += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Initialize with center point
        safe_eval(np.zeros(dim))

        # Phase 1: CMA-ES for global exploration
        cma_budget = int(budget * 0.6)
        lam = 2 * (dim + 1)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        x = best_x.copy()
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        while total_calls < cma_budget and time_ok():
            candidates = []
            for i in range(lam):
                if not time_ok() or total_calls >= cma_budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                fy = safe_eval(y)
                candidates.append((fy, y))

            if len(candidates) < mu:
                break

            candidates.sort(key=lambda t: t[0])
            y_best = np.array([c[1] for c in candidates[:mu]])
            m = np.sum(weights[:, None] * y_best, axis=0)

            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / old_sigma * (B @ (D * (m - x)))

            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * total_calls / dim)) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (m - x) / old_sigma
                delta_h = 0
            else:
                pc = (1 - c_c) * pc
                delta_h = 1

            C = B @ (D[:, None] * (D[None, :] * B.T))
            C_new = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)
            C_new = (C_new + C_new.T) / 2
            D2, B2 = np.linalg.eigh(C_new)
            D2 = np.maximum(D2, 1e-10)
            D = np.sqrt(D2)
            B = B2

            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e6)
            x = m

        # Phase 2: Restart-based local refinement
        # Use remaining budget for multiple local restarts around best_x
        remaining_budget = budget - total_calls
        if remaining_budget > 10 and time_ok():
            # Try Nelder-Mead from best_x
            try:
                res = minimize(safe_eval, best_x, method='Nelder-Mead', 
                               options={'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

            # If we have more budget, do a few more local restarts with different perturbations
            if total_calls < budget and time_ok():
                for attempt in range(3):
                    if total_calls >= budget or not time_ok():
                        break
                    # Perturb in random direction
                    perturbation_scale = 0.5 * (0.5 ** attempt)
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    start_point = best_x + perturbation_scale * direction
                    start_point = np.clip(start_point, -5, 5)
                    
                    try:
                        res2 = minimize(safe_eval, start_point, method='Nelder-Mead', 
                                       options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                    except Exception:
                        pass

        return best_x, best_f
