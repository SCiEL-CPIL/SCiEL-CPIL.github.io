"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal CMA-ES with stagnation-triggered restarts for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_diagonal_cma(self, func, rng, start_x, start_sigma, max_calls, deadline, 
                          lambda_=20, mu=None, restart_threshold=50):
        """
        Runs Diagonal CMA-ES.
        Returns best_x, best_fx, calls_made.
        """
        dim = self.dim
        if mu is None:
            mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State
        m = start_x.copy()
        sigma = start_sigma
        # Diagonal of covariance matrix, initialized to 1 (variance 1)
        diag_C = np.ones(dim)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = m.copy()
        best_fx = func(best_x)
        calls = 1
        gen = 0
        best_fx_at_restart_check = best_fx
        calls_at_restart_check = 0
        stagnation_count = 0

        while calls < max_calls and time.time() < deadline:
            # Determine population size for this generation
            remaining = max_calls - calls
            lam_cur = min(lambda_, remaining)
            if lam_cur < 2:
                break
            
            # Generate population
            ys = []
            fs = []
            for i in range(lam_cur):
                z = rng.standard_normal(dim)
                # y = m + sigma * sqrt(diag_C) * z
                # Note: In standard CMA-ES, y = m + sigma * B @ (D * z).
                # For diagonal, B=I, D=sqrt(diag_C).
                y = m + sigma * (np.sqrt(diag_C) * z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls += 1
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                
                if time.time() > deadline:
                    break
            
            if len(ys) < mu:
                break

            # Sort by fitness
            idx = np.argsort(fs)[:mu]
            ys_sorted = np.array([ys[i] for i in idx])
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * ys_sorted, axis=0)
            
            # Update evolution path ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((m - m_old) / sigma)))
            # For diagonal, B=I, D=sqrt(diag_C).
            # (m - m_old) is the step in x-space.
            # We need to map it back to the z-space equivalent for the path update?
            # Standard formula: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff)/sigma * B^T (m - m_old)
            # Since B=I, ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff)/sigma * (m - m_old)
            # Wait, the standard formula uses B^T (m - m_old). If C is diagonal, B=I.
            # However, the step size sigma scales the whole thing.
            # Let's stick to the standard implementation logic for diagonal.
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m - m_old)
            
            # Update sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - m_old)
            
            # Update diagonal covariance
            # diag_C_new = (1 + c_1 * delta_hsig - c_1 - c_mu) * diag_C + c_1 * pc^2 + c_mu * sum(w_i * (y_i - m_old)^2)
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            # Calculate weighted sum of squared differences
            diffs = ys_sorted - m_old
            weighted_sums = np.sum(w[:, None] * (diffs ** 2), axis=0)
            
            diag_C = (1 + c_1 * delta_hsig - c_1 - c_mu) * diag_C + c_1 * (pc ** 2) + c_mu * weighted_sums
            
            # Ensure positive semi-definiteness (diagonal elements must be positive)
            diag_C = np.maximum(diag_C, 1e-12)
            
            gen += 1
            
            # Stagnation check
            if calls - calls_at_restart_check >= restart_threshold:
                if best_fx < best_fx_at_restart_check - 1e-6:
                    # Improved, reset counters
                    best_fx_at_restart_check = best_fx
                    calls_at_restart_check = calls
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    calls_at_restart_check = calls
                    if stagnation_count > 3:
                        # Restart: break out of this CMA run
                        break
            else:
                calls_at_restart_check = calls

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        total_calls = 0
        best_x = None
        best_fx = np.inf

        # Phase 1: Initial diverse sampling to find a good starting point
        # Use a small number of random points to avoid starting in a bad basin
        n_init = min(10, self.budget)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            if time.time() > deadline:
                return best_x, best_fx

        # Phase 2: Iterative Diagonal CMA-ES with restarts
        # We will run multiple instances of Diagonal CMA-ES, restarting around the best found so far
        # if stagnation is detected or if the step size collapses.
        
        max_calls_per_run = 100  # Limit calls per CMA run to allow for restarts
        restarts_allowed = 4

        current_x = best_x.copy()
        current_sigma = 2.0

        for r in range(restarts_allowed):
            if total_calls >= self.budget or time.time() > deadline:
                break

            # Run Diagonal CMA-ES
            new_best_x, new_best_fx, calls_made = self._run_diagonal_cma(
                func, rng, current_x, current_sigma, 
                max_calls=self.budget - total_calls, 
                deadline=deadline,
                lambda_=20,
                restart_threshold=30
            )
            
            total_calls += calls_made
            
            if new_best_fx < best_fx:
                best_fx = new_best_fx
                best_x = new_best_x.copy()
                # If we made significant progress, continue refining around the new best
                # with a slightly smaller or same sigma
                current_x = new_best_x.copy()
                # Keep sigma moderate for next restart if it happens
                current_sigma = 1.0 
            else:
                # No improvement, try a larger step size or random restart
                current_x = best_x.copy()
                current_sigma = 3.0 + r * 0.5

            # If we are very close to convergence (sigma very small), break
            # We don't have direct access to final sigma, but if calls_made is low, it converged fast.
            
            if time.time() > deadline:
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
