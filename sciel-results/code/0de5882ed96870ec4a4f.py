"""
Hybrid optimizer for BBOB f3 (Rastrigin): CMA-ES + Nelder-Mead local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        x_mean = np.zeros(n)
        sigma = 1.5
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def time_check():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: CMA-ES
        generation = 0
        while not time_check():
            generation += 1
            ys = []
            for k in range(lam):
                z = np.random.standard_normal(n)
                y = B @ (D * z)
                ys.append(y)
            
            xs = x_mean + sigma * np.array(ys)
            
            remaining = budget - calls_made
            num_evals = min(lam, remaining)
            if num_evals <= 0:
                break
            
            fitnesses = []
            eval_xs = []
            for i in range(num_evals):
                x = xs[i]
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                fitnesses.append(f)
                eval_xs.append(x)
                if calls_made >= budget:
                    break
            
            if calls_made >= budget or num_evals < lam:
                break
                
            fitnesses = np.array(fitnesses)
            eval_xs = np.array(eval_xs)
            
            order = np.argsort(fitnesses)
            xs_sorted = eval_xs[order]
            fitnesses_sorted = fitnesses[order]
            
            new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            y_mean = (new_mean - x_mean) / sigma
            
            c_hs = (1.1 + 2.0 / (n + 1.7))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
            
            norm_y_mean = np.linalg.norm(y_mean)
            sqrt_trace_C = np.sqrt(np.trace(C))
            
            ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * generation)) < (1.4 + 2 / (n + 1)) * chi_n)
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            ys_array = np.array(ys)
            ys_sorted = ys_array[order][:mu]
            
            rank_one = np.outer(p_c, p_c)
            
            y_diffs = ys_sorted - y_mean
            rank_mu = np.sum(weights[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            C = (C + C.T) / 2
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
            
            x_mean = new_mean
            
            if sigma < 1e-12:
                break

        # Phase 2: Local refinement with Nelder-Mead
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            # Use Nelder-Mead for local refinement
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, remaining // 2),
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final coordinate descent polish
        if best_x is not None and calls_made < budget:
            step = 0.05
            for _ in range(50):
                if calls_made >= budget:
                    break
                improved = False
                for i in range(n):
                    for direction in [1, -1]:
                        if calls_made >= budget:
                            break
                        x_new = best_x.copy()
                        x_new[i] += direction * step
                        if np.all(x_new >= -5.0) and np.all(x_new <= 5.0):
                            f_new = func(x_new)
                            calls_made += 1
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                                improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
