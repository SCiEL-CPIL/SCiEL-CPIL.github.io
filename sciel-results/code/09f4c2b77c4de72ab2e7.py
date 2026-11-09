"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Fast CMA-ES with radial diversity initialization for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 270  # Safety margin vs 300s limit

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting (standard for dim=5) ----
        lambda_ = 14
        mu_ = 7
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # ---- CMA-ES strategy constants ----
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ---- state ----
        x_mean = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        L = np.eye(dim)          # Cholesky: C = L L^T
        have_L = True

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(y)
            call_count += 1
            if f < best_f:
                best_f = float(f)
                best_x = y.copy()
            return f

        # ---- Initial Diverse Exploration: Radial Diversity ----
        # f11 (Discus) has a ring of minima at a specific radius.
        # We initialize points at various radii in random directions to
        # help CMA-ES quickly identify the correct scale and location.
        n_init = 16
        # Radii to sample: cover the likely range of the optimum (norm ~ 2-4)
        target_radii = [1.0, 2.0, 3.0, 4.0]
        
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            
            # Select a target radius, cycling through the list
            r_target = target_radii[k % len(target_radii)]
            
            # Generate a random direction
            direction = rng.standard_normal(dim)
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
            else:
                direction = np.array([1.0] + [0.0]*(dim-1))
                
            x0 = r_target * direction
            
            # Add small jitter to avoid exact overlaps if directions collide
            x0 += 0.05 * rng.standard_normal(dim)
            
            x0 = np.clip(x0, -5.0, 5.0)
            evaluate(x0)

        # ---- Main CMA-ES Loop ----
        # Reserve some budget for the final polish step
        polish_reserved = 20
        cma_budget = budget - polish_reserved
        
        while call_count < cma_budget:
            if timed_out():
                break
            
            # Generate population
            offspring = []
            for _ in range(lambda_):
                if call_count >= cma_budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                f = evaluate(y)
                offspring.append((f, y, z))

            if len(offspring) < mu_:
                break

            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu_]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
            delta = x_mean - old_mean

            # Evolution paths
            if have_L:
                try:
                    Cinv_half_delta = np.linalg.solve(L.T, np.linalg.solve(L, delta))
                except np.linalg.LinAlgError:
                    have_L = False
                    Cinv_half_delta = delta
            else:
                Cinv_half_delta = delta

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * Cinv_half_delta

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 20.0))

            # Direct covariance update
            drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
            C = (1 - c_1 - c_mu) * (L @ L.T)
            C += c_1 * (drift * np.outer(p_c, p_c) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim))
            for i in range(mu_):
                d = selected[i][1] - x_mean
                C += c_mu * w[i] * np.outer(d, d)
            C = 0.5 * (C + C.T)

            # Refresh Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                have_L = True
            except np.linalg.LinAlgError:
                L = L + 1e-4 * np.eye(dim)
                have_L = False

            # Stagnation guard
            if sigma < 1e-7:
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                sigma = 0.5

        # ---- Final Polish Pass ----
        # Use Nelder-Mead for high-precision refinement around the best point found
        if best_x is not None and call_count < budget and not timed_out():
            polish_budget = min(20, budget - call_count)
            if polish_budget > 5:
                polish_calls = [0]
                def polish_func(x):
                    if polish_calls[0] >= polish_budget or timed_out():
                        return best_f
                    y = np.clip(x, -5.0, 5.0)
                    f = func(y)
                    polish_calls[0] += 1
                    if f < best_f:
                        best_f = float(f)
                        best_x = y.copy()
                    return f
                
                try:
                    # Use a tight tolerance to maximize precision in remaining calls
                    res = minimize(
                        polish_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
