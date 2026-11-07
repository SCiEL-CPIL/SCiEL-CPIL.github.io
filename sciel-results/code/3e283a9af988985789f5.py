"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        best_x = None
        best_f = float('inf')
        total_budget = self.budget
        restarts = 4
        budget_per_restart = total_budget // restarts
        remaining = total_budget

        for restart in range(restarts):
            if remaining <= 0:
                break
            
            # For the first restart, use a reasonable initial guess
            if restart == 0:
                # Start near center, but slightly offset to explore
                x = np.zeros(self.dim)
                # Small random perturbation
                x += np.random.default_rng(42 + restart * 100).uniform(-0.5, 0.5, self.dim)
            else:
                # For subsequent restarts, start near the current best
                if best_x is not None:
                    x = best_x.copy()
                    # Add some randomness to escape local optima
                    rng = np.random.default_rng(123 + restart * 100)
                    x = x + rng.uniform(-0.5, 0.5, self.dim)
                    x = np.clip(x, -5, 5)
                else:
                    rng = np.random.default_rng(123 + restart * 100)
                    x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            remaining -= 1
            if remaining <= 0:
                break
            
            # CMA-ES-like parameters
            dim = self.dim
            # Population size
            lam = max(4 + int(3 * np.log(dim)), 6)
            # Step size
            sigma = 1.0
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            # Learning rates
            cs = (lam + 2) / (dim + lam + 4)
            c1 = 2 / ((dim + 1.3) ** 2 + lam)
            cmu = min(1 - c1, 2 * ((lam - 2 + 1 / lam) / ((dim + 2) ** 2 + lam) - 1))
            # Damping for pc
            damps = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1) - 1) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # History of best points
            history = []
            
            iters = remaining
            for it in range(iters):
                if remaining <= 0:
                    break
                
                # Generate population
                Y = []
                for j in range(lam):
                    rng = np.random.default_rng(12345 + restart * 1000 + it * 100 + j)
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                
                # Evaluate
                FY = [func(y) for y in Y]
                for y, fy in zip(Y, FY):
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    remaining -= 1
                    if remaining <= 0:
                        break
                
                if remaining <= 0:
                    break
                
                # Sort by fitness
                indices = np.argsort(FY)
                Y_sorted = [Y[i] for i in indices]
                FY_sorted = [FY[i] for i in indices]
                
                # Update mean
                mu = lam // 2
                weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights ** 2)
                
                x_old = x.copy()
                x = np.sum([weights[i] * Y_sorted[i] for i in range(mu)], axis=0)
                x = np.clip(x, -5, 5)
                
                # Update evolution paths
                if np.linalg.norm(x - x_old) > 1e-10:
                    z_bar = (x - x_old) / (sigma * np.linalg.norm(x - x_old))
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_bar
                    hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) <= (1.4 + 2 / (dim + 1)) * chi_n
                    if hsig:
                        pc = (1 - c1) * pc + c1 * (z_bar @ z_bar.T - np.eye(dim)) * 0  # Simplified
                        # Actually, proper update:
                        pc = (1 - c1) * pc + c1 * (np.outer(z_bar, z_bar) - np.eye(dim))
                    else:
                        pc = (1 - c1) * pc
                    
                    # Update covariance
                    C = (1 - c1 - cmu) * C
                    C += c1 * (np.outer(ps, ps) + (2 * hsig - 1) * pc + np.eye(dim) * 0)
                    # Simpler: C = (1-c1-cmu)*C + c1*(ps@ps.T + (2*hsig-1)*pc) + cmu * sum(w_i * (y_i-x_old)(y_i-x_old).T)
                    # Let's use a simpler covariance update
                    C = (1 - c1 - cmu) * C
                    C += c1 * (np.outer(ps, ps) + (2 * hsig - 1) * pc)
                    for i in range(mu):
                        diff = (Y_sorted[i] - x_old) / sigma
                        C += cmu * weights[i] * np.outer(diff, diff)
                    
                    # Ensure C is positive definite
                    C = (C + C.T) / 2
                    # Add small regularization
                    C += 1e-8 * np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # If we've converged, break early and restart
                if it > 20 and np.std(FY_sorted[:mu]) < 1e-8:
                    break
            
            if remaining <= 0:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
