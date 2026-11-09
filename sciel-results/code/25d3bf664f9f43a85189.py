"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer with adaptive restarts for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._start_time = None

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 280.0  # Safety margin before 300s limit

        # Initial random evaluation to have a valid first call
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4, 4, self.dim)
        f0 = float(func(x0))
        self._record(x0, f0)

        if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
            return self._best_x, self._best_f

        # CMA-ES Parameters
        # Popsize: 4 + floor(3 * ln(n)) is standard. For n=5, 4+3*1.6=8.8 -> 9.
        # Let's use a slightly larger population for better exploration initially, 
        # but keep it efficient. lambda = 10 is a good balance for dim=5.
        lam = 10
        
        # Initial sigma
        sigma = 2.0
        
        # CMA-ES state
        best_x = self._best_x.copy()
        sigma = 2.0
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        C_sqrt = np.eye(self.dim)
        
        # Adaptation parameters
        # Standard CMA-ES parameters
        cmu = 2 / ((self.dim + 4)**1.5 + 10)
        c_sigma = (mu_eff + 2) / ((self.dim + 1)**1.5 + mu_eff)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff/(self.dim+1)) - 1) + c_sigma
        c_c = (4 + mu_eff/self.dim) / (self.dim + 4 + 2*mu_eff/self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (cmu + 2/self.dim) / ((self.dim + 2)**2 + mu_eff))
        
        mu_eff = lam / 2
        mu = np.zeros(lam)
        # Weights: log-linear weights
        w = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
        w = w / w.sum()
        mu_eff = 1 / sum(w**2)
        
        # Initialize C and C_inv
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        C_sqrt = np.eye(self.dim)
        
        # Cumulation paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Restart logic
        stagnation_count = 0
        best_f_restart = self._best_f
        restart_count = 0
        max_stagnation = 50  # Generations without improvement before increasing sigma
        max_stagnation_restart = 150 # Generations without improvement before full restart

        while self._eval_count < self.budget and (time.time() - self._start_time) < deadline:
            
            # Generate offspring
            # Update C_sqrt if C has changed (we will update C after each generation)
            # For simplicity and speed, we can update C_sqrt every generation using eigendecomposition
            # Or we can use a simpler approximation. Given dim=5, eigendecomposition is fast.
            
            # Generate random normals
            z = rng.normal(0, 1, (lam, self.dim))
            
            # Transform to search space
            # x = mean + sigma * C_sqrt @ z
            # We need C_sqrt. Let's compute it.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Ensure positive definite
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            
            mean = self._best_x if self._best_x is not None else np.zeros(self.dim)
            # Actually, we should maintain a separate 'mean' for CMA-ES, 
            # but using the current best as the mean is a common heuristic for restarts.
            # Standard CMA-ES updates the mean. Let's maintain a proper mean.
            
            # Let's implement a more standard CMA-ES loop structure.
            
            # --- Standard CMA-ES Step ---
            
            # 1. Sample
            z = rng.normal(0, 1, (lam, self.dim))
            # We need C_sqrt. 
            # Optimization: Only compute C_sqrt every few iterations or if C changes significantly.
            # For dim=5, it's cheap enough to do every iteration.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            
            y = z @ C_sqrt.T
            x = mean + sigma * y
            
            # Evaluate
            fx = np.array([float(func(xi)) for xi in x])
            for i in range(lam):
                self._record(x[i], fx[i])
                
            if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
                return self._best_x, self._best_f

            # Sort by fitness
            idx = np.argsort(fx)
            x_sorted = x[idx]
            fx_sorted = fx[idx]
            
            # 2. Update Mean
            w_mu = w[:mu_eff]
            mean_new = sum(w_mu[i] * x_sorted[i] for i in range(mu_eff))
            
            # 3. Update Covariance Matrix and Step Size
            # Compute cumulation paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt @ (mean - mean_new).T
            # Wait, the formula is:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * C^{-1/2} (mean - mean_new)
            
            # We need C_inv_sqrt.
            C_inv_sqrt = (eigvecs * (1/np.sqrt(eigvals))) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (mean - mean_new).T).T
            
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(self._eval_count//lam + 1))) / np.sqrt(mu_eff)
            # Approximation for h_sigma check
            h_sigma = 1 if np.linalg.norm(p_sigma) < (1.4 + 2/(self.dim+1)) * np.sqrt(mu_eff) else 0
            
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) / sigma * (mean - mean_new).T
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean - mean_new).T
            
            # Update C
            # C = (1 - c_1 - c_mu + h_sigma c_1 (2 - c_c)) C + c_1 (p_c p_c^T - C) + c_mu sum w_i (y_i y_i^T)
            y_sorted = (x_sorted[:mu_eff] - mean) / sigma
            C = (1 - c_1 - c_mu + h_sigma * c_1 * (2 - c_c)) * C + c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu_eff):
                C += c_mu * w_mu[i] * np.outer(y_sorted[i], y_sorted[i])
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-6, 100)
            
            mean = mean_new
            
            # 4. Stagnation Check
            if self._best_f < best_f_restart - 1e-8:
                best_f_restart = self._best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Adaptive Sigma Increase
            if stagnation_count > max_stagnation:
                sigma *= 1.5
                stagnation_count = 0 # Reset to give it another chance
            
            # Restart
            if stagnation_count > max_stagnation_restart:
                # Restart from best point with larger sigma
                mean = self._best_x.copy()
                sigma = 2.0
                C = np.eye(self.dim)
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)
                stagnation_count = 0
                best_f_restart = self._best_f
                restart_count += 1
                
                # If we restart too many times, we might be stuck.
                # But with adaptive sigma, it should eventually escape or run out of budget.
                
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
