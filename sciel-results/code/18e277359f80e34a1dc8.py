"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-phase CMA-ES optimizer for the BBOB Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 300.0  # Safety margin before 330s kill

        # Multi-phase strategy: progressively smaller sigmas
        phases = [4.0, 1.0, 0.2, 0.05]
        
        best_x = np.zeros(dim)
        best_fx = np.inf
        call_count = 0

        for sigma_init in phases:
            if call_count >= budget:
                break
            if time.time() - start_time > timeout:
                break
            
            # Run CMA-ES with this initial sigma
            x, fx, calls = self._run_cmaes(func, best_x.copy(), sigma_init, dim, budget - call_count, start_time, timeout)
            call_count += calls
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        return best_x, best_fx

    def _run_cmaes(self, func, x0, sigma0, dim, max_calls, start_time, timeout):
        """Run a single CMA-ES instance."""
        x = x0.copy()
        sigma = sigma0
        fx = func(x)
        calls = 1
        
        # CMA-ES parameters for dim=5
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_fx = fx
        best_x = x.copy()
        stagnation_counter = 0
        stagnation_threshold = 20  # generations without improvement
        
        while calls < max_calls:
            if time.time() - start_time > timeout:
                break
            
            # Check if sigma is too small
            if sigma < 1e-12:
                break
                
            # Generate population
            z = np.random.randn(lam, dim)
            children = x[None, :] + sigma * (z @ C_sqrt.T)
            children = np.clip(children, -5, 5)
            
            # Evaluate
            f_children = np.array([func(child) for child in children])
            calls += lam
            
            # Sort by fitness
            order = np.argsort(f_children)
            y_sorted = children[order]
            f_sorted = f_children[order]
            
            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = y_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Check stagnation
            if stagnation_counter > stagnation_threshold:
                break
                
            # Mean of selected individuals
            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update p_sigma
            y_sigma = (m_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = min(1, np.linalg.norm(p_sigma) / (1 - (1 - c_sigma)**((lam + 1) / 2)) / chi_n)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            C = (1 + c_1 + c_mu - c_c * (1 - c_c) - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * np.sum(weights[:, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update x
            x = m_new
            
            # Update C_sqrt via Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add small regularization
                C += 1e-10 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break
        
        return best_x, best_fx, calls

# EVOLVE-BLOCK-END
