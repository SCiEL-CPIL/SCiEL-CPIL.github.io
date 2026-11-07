"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polish and (1+1)-ES fallback."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # 300s limit, 25s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: CMA-ES from center
        if calls[0] < budget:
            try:
                x0 = np.zeros(dim)
                sigma0 = 2.0
                pop_size = min(12, max(4, (budget - calls[0]) // 10))
                
                # Manual CMA-ES for better control
                x = x0.copy()
                sigma = sigma0
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                
                mu = pop_size // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights**2)
                
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
                
                iter_count = 0
                max_iters = min(200, (budget - calls[0]) // max(pop_size, 1))
                
                for it in range(max_iters):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    
                    # Generate population
                    offspring = []
                    for i in range(pop_size):
                        z = rng.standard_normal(dim)
                        y = x + sigma * (B @ (D * z))
                        y = np.clip(y, -5, 5)
                        offspring.append(y)
                    
                    # Evaluate
                    fvals = []
                    for y in offspring:
                        try:
                            fy = wrapped(y)
                        except Exception:
                            fy = np.inf
                        fvals.append(fy)
                    
                    fvals = np.array(fvals)
                    
                    # Sort by fitness
                    order = np.argsort(fvals)
                    offspring_sorted = [offspring[i] for i in order]
                    fvals_sorted = fvals[order]
                    
                    # Update best
                    for i in range(min(mu, len(fvals_sorted))):
                        if fvals_sorted[i] < np.inf:
                            update_best(offspring_sorted[i], fvals_sorted[i])
                    
                    # Update mean
                    x_new = sum(weights[i] * offspring_sorted[i] for i in range(mu))
                    
                    # Update step size
                    if np.linalg.norm(B.T @ (D * (ps / sigma))) > 1.4 * chi_N + 1:
                        sigma *= np.exp(c_sigma / c_c * (np.linalg.norm(B.T @ (D * (ps / sigma))) / chi_N - 1))
                    
                    # Update evolution paths
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (D * ((x_new - x) / sigma)))
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                    
                    # Update covariance matrix
                    delta_hs = (1 + 2 * max(0, np.linalg.norm(ps) / (1 - np.sqrt(c_c)) - 1)) / chi_N
                    C = (1 - c_1 - c_mu) * C
                    C += c_1 * (np.outer(pc, pc) - C)
                    for i in range(mu):
                        y_i = (B.T @ (D * ((offspring_sorted[i] - x) / sigma)))
                        C += c_mu * weights[i] * np.outer(y_i, y_i)
                    
                    # Recompute B and D
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                    
                    x = x_new
                    
                    # Check convergence
                    if sigma < 1e-10 or np.all(np.abs(x) < 1e-8):
                        break
                    
                    iter_count += 1
                
            except Exception:
                pass

        # Phase 2: L-BFGS-B from best point
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 30, 'gtol': 1e-8, 'ftol': 1e-14}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: Try a few random restarts with L-BFGS-B if budget allows
        if calls[0] < budget and time.time() < deadline:
            n_restarts = min(5, max(1, (budget - calls[0]) // 30))
            for _ in range(n_restarts):
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    s = rng.uniform(-4, 4, dim)
                    res = minimize(
                        wrapped,
                        s,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxls': 20, 'gtol': 1e-6, 'ftol': 1e-12}
                    )
                    update_best(res.x, res.fun)
                except Exception:
                    pass

        # Phase 4: (1+1)-ES polish from best point
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if sigma < 1e-8:
                    break

        # Phase 5: Nelder-Mead final polish
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                nm_maxiter = min(30, max(5, (budget - calls[0]) // 3))
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
