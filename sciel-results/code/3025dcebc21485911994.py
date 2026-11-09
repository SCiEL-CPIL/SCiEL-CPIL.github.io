"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
CMA-ES with restarts, leveraging covariance adaptation for rotated landscapes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global state
        best_x = None
        best_f = np.inf
        eval_count = 0

        class _BudgetExceeded(Exception):
            pass

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            try:
                f_val = float(func(np.asarray(x, dtype=np.float64)))
            except Exception:
                f_val = 1e10
            if not np.isfinite(f_val):
                f_val = 1e10
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Initial sampling to locate basins
        n_samples = min(20, budget)
        # Try origin and structured points
        structured = [np.zeros(n)]
        for offset in [1.0, -1.0, 2.0, -2.0, 3.5, -3.5]:
            structured.append(np.full(n, offset))
        
        for pt in structured[:n_samples]:
            if eval_count >= budget or time.time() > deadline:
                break
            safe_func(pt)
        
        # Random samples for coverage
        while eval_count < n_samples and eval_count < budget:
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)

        if best_f == np.inf or best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: CMA-ES with restarts
        max_restarts = 5
        restart_count = 0
        no_improve_count = 0
        no_improve_threshold = 150  # evaluations without improvement triggers restart consideration

        while restart_count < max_restarts and eval_count < budget and time.time() < deadline:
            # Initialize CMA-ES parameters
            x_mean = best_x.copy()
            f_best = best_f
            
            # Initial sigma based on distance from previous best
            if restart_count == 0:
                sigma = 1.0
            else:
                sigma = 0.5

            # CMA-ES parameters
            lam = 4 * n  # population size
            mu = lam // 2  # number of parents

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)

            chi_n = np.sqrt(n) * (1 - 1.0/(4*n) - 1.0/(20*n**2))
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
            cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)

            # Covariance matrix and evolution paths
            C = np.eye(n)
            Cs = np.zeros(n)
            ps = np.zeros(n)
            
            # Cholesky factor of C
            try:
                C_eigen = np.linalg.eigh(C)
                C_eigenvalues = C_eigen[0]
                C_eigenvalues = np.maximum(C_eigenvalues, 1e-20)
                C_eigenvecs = C_eigen[1]
                C_cholesky = C_eigenvecs @ np.diag(np.sqrt(C_eigenvalues))
            except np.linalg.LinAlgError:
                C_cholesky = np.eye(n)

            f_prev = f_best
            stall_count = 0
            max_stall = 100  # max evaluations without improvement before forcing restart

            while eval_count < budget and time.time() < deadline:
                # Generate population
                Y = []
                F = []
                
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_cholesky @ z)
                    y = np.clip(y, -5.0, 5.0)
                    Y.append(y)
                    
                    try:
                        fy = safe_func(y)
                    except _BudgetExceeded:
                        raise
                    F.append(fy)

                if len(F) == 0:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                Y_sorted = [Y[i] for i in idx]
                F_sorted = [F[i] for i in idx]
                
                # Update best
                if F_sorted[0] < best_f:
                    best_f = F_sorted[0]
                    best_x = Y_sorted[0].copy()
                    stall_count = 0
                else:
                    stall_count += 1

                # Check for stall
                if stall_count > max_stall:
                    break

                # Update mean
                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += w[i] * Y_sorted[i]
                
                # Update evolution paths
                d = x_mean_new - x_mean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * d
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) / chi_n <= (1 + 2 / (n + 1))
                Cs = (1 - ds) * Cs + hsig * np.sqrt(ds * (2 - ds) * mu_eff) * (d / sigma)

                # Update covariance
                delta_hsig = (1 - hsig) * ds + 1
                C = (1 + cmu * delta_hsig - cmu - cs * (1 - cs)) * C
                C += cmu * np.outer(d, d) / sigma**2
                C += cs * ds * (1 - cs) * np.outer(Cs, Cs)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                # Update Cholesky
                try:
                    C_eigen = np.linalg.eigh(C)
                    C_eigenvalues = C_eigen[0]
                    C_eigenvalues = np.maximum(C_eigenvalues, 1e-20)
                    C_eigenvecs = C_eigen[1]
                    C_cholesky = C_eigenvecs @ np.diag(np.sqrt(C_eigenvalues))
                except np.linalg.LinAlgError:
                    C_cholesky = np.eye(n)

                x_mean = x_mean_new

            # After CMA-ES run, check if we should restart
            restart_count += 1

            # If we found a better solution, continue with new center
            if best_x is not None:
                # Reset for next restart with smaller sigma
                pass

        # Phase 3: Local refinement with (1+1)-ES
        if eval_count < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            
            # (1+1)-ES with 1/5 rule
            successes = 0
            total = 0
            
            while eval_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
                
                try:
                    fy = safe_func(y)
                except _BudgetExceeded:
                    break
                
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.15
                else:
                    sigma *= 0.95
                
                sigma = max(1e-8, min(sigma, 1.0))
                
                # Check for convergence
                if total >= 30:
                    success_rate = successes / total
                    if success_rate < 0.05 or success_rate > 0.4:
                        successes = 0
                        total = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
