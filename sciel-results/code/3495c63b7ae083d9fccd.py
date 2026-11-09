"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance matrix for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Random sampling to find a good initial basin
        # f18 is shifted/rotated. A few random points help avoid starting in a bad region.
        n_random = min(20, budget)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, n)
            call_func(x)

        if best_x is None:
            return np.zeros(n), np.inf

        # Phase 2: CMA-ES
        # Parameters
        N = 4 + int(3 * np.log(n))  # Population size
        if N > budget - calls:
            N = max(2, budget - calls)
        
        lambda_ = N
        mu = lambda_ // 2  # Number of selected solutions
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size. 1.0 is reasonable for 5D box.
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute
        chi_n = n * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # B and D for sampling
        # We will recompute B and D every generation or use eigendecomposition
        # For efficiency in small n, we can use np.linalg.eigh
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if sigma is too small or variance is too small
            # Simple check: if sigma < 1e-6, we are likely converged or stuck
            if sigma < 1e-6:
                # Try a random restart if we have budget
                if calls + 10 < budget:
                    x_new = rng.uniform(-4, 4, n)
                    f_new = call_func(x_new)
                    if f_new < best_f:
                        x_mean = x_new
                        sigma = 1.0
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                    else:
                        break
                else:
                    break

            # Generate population
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            Y = rng.standard_normal((N, n))
            X = np.array([x_mean + sigma * (Y[i] @ B * D) for i in range(N)])
            
            # Evaluate
            F = np.array([call_func(X[i]) for i in range(N)])
            
            # Sort indices by fitness
            idx = np.argsort(F)
            y_selected = np.array([X[idx[i]] for i in range(mu)])
            
            # Update mean
            x_new_mean = np.sum(w[:, None] * y_selected, axis=0)
            
            # Update step size (CSA)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D * ((pc @ B) / sigma))
            # Note: The standard update is:
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * B * (D^-1 * (B^T * (x_new_mean - x_mean)/sigma))
            
            y_mean_diff = (x_new_mean - x_mean) / sigma
            # B @ (D * (B^T @ y_mean_diff)) is equivalent to C @ y_mean_diff
            # But we need B @ (D * (B^T @ y_mean_diff)) / sigma? No.
            # Formula: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B @ (D * (B^T @ (x_new_mean - x_mean) / sigma)))
            
            term = B @ (D * (B.T @ y_mean_diff))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * term
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean_diff
            
            # Rank-1 update
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - budget + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            # Simplified h_sigma check for small n
            # Actually, standard h_sigma is:
            # h_sigma = 1 if norm(ps) < (1.4 + 2/(n+1)) * chi_n * sqrt(1 - (1-c_sigma)^(2*gen))
            # Since we don't track gen precisely in this loop without a counter, we use a simpler heuristic or approximate.
            # For robustness, we can just use the standard formula with a dummy gen or approximate.
            # Let's use a simple threshold check.
            h_sigma = 1.0 if np.linalg.norm(ps) < 1.5 * chi_n else 0.0
            
            C = (1 - c_1 - c_mu + (1 - h_sigma) * c_1 * (2 - c_c)) * C + \
                c_1 * np.outer(pc, pc) + \
                c_mu * np.sum(w[i] * np.outer(y_selected[i] - x_mean, y_selected[i] - x_mean) for i in range(mu)) / sigma**2
            
            x_mean = x_new_mean
            
            # Termination criteria
            # If the population has collapsed
            if sigma < 1e-6:
                break
            # If the best fitness hasn't improved in many generations, we might be stuck.
            # But CMA-ES is robust. Let's rely on sigma collapse or budget.

        return best_x, best_f
# EVOLVE-BLOCK-END
