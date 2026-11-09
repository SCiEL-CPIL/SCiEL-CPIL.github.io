import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with stagnation restart and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        # CMA-ES Parameters
        lam = 12  # Population size
        mu = lam // 2  # Number of parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        eval_count = 1

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        evals_since_improvement = 0
        stagnation_threshold = 36  # 3 generations of 12

        def cma_es_step():
            nonlocal x, sigma, C, pc, ps, evals_since_improvement, eval_count
            
            # Sample
            z = rng.standard_normal((lam, dim))
            # Cholesky of C
            L = np.linalg.cholesky(C)
            y = (z @ L.T) * sigma
            X = x[None, :] + y
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            eval_count += lam
            
            # Sort
            idx = np.argsort(F)
            y_sorted = y[idx]
            F_sorted = F[idx]
            X_sorted = X[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                nonlocal best_f, best_x
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                evals_since_improvement = 0
            else:
                evals_since_improvement += lam

            # Update mean
            x_new = x + sigma * (w[:mu] @ y_sorted[:mu])
            
            # Update evolution paths
            # Norm of B.T @ sqrt(D) is needed for ps update
            # We maintain B, D from previous C decomposition
            # But C changes, so we need to update B, D first? 
            # Standard CMA-ES updates C, then decomposes for next step.
            # The path update uses the *current* B, D (from previous C).
            
            # We need B, D from the *previous* C.
            # I will store B, D from the end of the previous step.
            
            # Initialize B, D if not set
            if not hasattr(cma_es_step, 'B_prev'):
                cma_es_step.B_prev = np.eye(dim)
                cma_es_step.D_prev = np.ones(dim)
            
            B_prev = cma_es_step.B_prev
            D_prev = cma_es_step.D_prev
            
            # ps update
            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) / np.linalg.norm(B_prev.T @ np.sqrt(D_prev)) * (B_prev.T @ np.sqrt(D_prev) @ (w[:mu] @ y_sorted[:mu]))
            
            hs = np.linalg.norm(ps) / np.sqrt((1.0 - (1.0 - c_sigma) ** 2)) / np.linalg.norm(B_prev.T @ np.sqrt(D_prev))
            
            # pc update
            if hs < 1.4 + c_c:
                pc = (1.0 - c_c) * pc + np.sqrt(c_c * (2.0 - c_c) * mu_eff) / np.linalg.norm(B_prev.T @ np.sqrt(D_prev)) * (B_prev.T @ np.sqrt(D_prev) @ (x_new - x) / sigma)
            
            # Update covariance
            delta_h = (1.0 - c_c) ** 2 - (1.0 - c_c)
            rank_one = np.outer(pc, pc)
            # rank_mu
            y_scaled = y_sorted[:mu] / sigma
            rank_mu = (w[:mu] * y_scaled[:, None]) @ y_scaled[None, :]
            
            C = (1.0 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Decompose new C for next step
            try:
                B_new, D_new = np.linalg.eigh(C)
                D_new = np.maximum(D_new, 1e-10)
                cma_es_step.B_prev = B_new
                cma_es_step.D_prev = D_new
            except:
                C = np.eye(dim)
                cma_es_step.B_prev = np.eye(dim)
                cma_es_step.D_prev = np.ones(dim)

            x = x_new

        # Main Loop
        # Reserve 50 evals for polish
        polish_budget = 50
        while eval_count < budget - polish_budget and time.time() < deadline:
            cma_es_step()
            if evals_since_improvement > stagnation_threshold:
                # Restart
                x = best_x.copy()
                C = np.eye(dim)
                sigma = max(sigma, 1.5)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                cma_es_step.B_prev = np.eye(dim)
                cma_es_step.D_prev = np.ones(dim)
                evals_since_improvement = 0

        # Polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 10 and time.time() < deadline:
            calls = [0]
            def func_counted(x):
                calls[0] += 1
                if calls[0] > remaining:
                    return best_f
                return func(x)
            
            try:
                res = minimize(func_counted, best_x, method='Nelder-Mead', 
                               options={'maxiter': remaining // 2, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except:
                pass

        return best_x, best_f
