"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES with Quadratic Model Refinement for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s kill

        best_x = np.zeros(dim)
        best_fx = np.inf
        call_count = 0

        # Phase 1: CMA-ES for global exploration and coarse convergence
        # Use 60% of budget for CMA-ES
        cmaes_budget = int(budget * 0.6)
        x, fx, calls = self._run_cmaes(
            func, 
            np.zeros(dim), 
            sigma_init=3.0, 
            max_calls=cmaes_budget,
            start_time=start_time, 
            timeout=timeout
        )
        
        call_count += calls
        if fx < best_fx:
            best_fx = fx
            best_x = x.copy()

        # Phase 2: Quadratic Model-Based Local Refinement
        # Use the remaining 40% of budget
        remaining_budget = budget - call_count
        if remaining_budget > 0:
            x2, fx2, calls2 = self._run_quadratic_refinement(
                func, 
                best_x.copy(), 
                max_calls=remaining_budget,
                start_time=start_time, 
                timeout=timeout
            )
            call_count += calls2
            if fx2 < best_fx:
                best_fx = fx2
                best_x = x2.copy()

        return best_x, best_fx

    def _run_cmaes(self, func, x0, sigma0, max_calls, start_time, timeout):
        """Run a single CMA-ES instance."""
        dim = self.dim
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
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_fx = fx
        best_x = x.copy()
        stagnation_counter = 0
        stagnation_threshold = 15  # Generations without improvement
        
        while calls < max_calls:
            if time.time() - start_time > timeout:
                break
            
            # Check if sigma is too small
            if sigma < 1e-10:
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

    def _run_quadratic_refinement(self, func, x0, max_calls, start_time, timeout):
        """
        Local refinement using a diagonal quadratic model (Newton's method with central differences).
        """
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        best_fx = fx
        best_x = x.copy()

        # Initial step size for finite differences
        h = 0.5 
        min_h = 1e-6
        
        while calls < max_calls and h > min_h:
            if time.time() - start_time > timeout:
                break
            
            # Need 2n evaluations (plus the center which we already have)
            if calls + 2 * dim > max_calls:
                break
            
            f_plus = np.zeros(dim)
            f_minus = np.zeros(dim)
            
            for i in range(dim):
                # x + h * e_i
                x_plus = x.copy()
                x_plus[i] += h
                x_plus[i] = np.clip(x_plus[i], -5, 5)
                f_plus[i] = func(x_plus)
                calls += 1
                
                # x - h * e_i
                x_minus = x.copy()
                x_minus[i] -= h
                x_minus[i] = np.clip(x_minus[i], -5, 5)
                f_minus[i] = func(x_minus)
                calls += 1
                
                # Update best if any of these are better
                if f_plus[i] < best_fx:
                    best_fx = f_plus[i]
                    best_x = x_plus.copy()
                if f_minus[i] < best_fx:
                    best_fx = f_minus[i]
                    best_x = x_minus.copy()

            # Estimate gradient and diagonal Hessian
            g = (f_plus - f_minus) / (2 * h)
            hessian_diag = (f_plus + f_minus - 2 * fx) / (h * h)
            
            # Ensure Hessian is positive definite (add small regularization if needed)
            hessian_diag = np.maximum(hessian_diag, 1e-8)
            
            # Newton step: dx = -g / hessian_diag
            dx = -g / hessian_diag
            
            # Limit step size to prevent overshooting
            # If step is too large, scale it down
            max_step = h * 2
            norm_dx = np.linalg.norm(dx)
            if norm_dx > max_step:
                dx = dx * (max_step / norm_dx)
            
            x_new = x + dx
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                # Accept step
                x = x_new
                fx = f_new
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Shrink step size slightly for next iteration
                h *= 0.8
            else:
                # Reject step, shrink step size more aggressively
                h *= 0.5
                
        return best_x, best_fx, calls

# EVOLVE-BLOCK-END
