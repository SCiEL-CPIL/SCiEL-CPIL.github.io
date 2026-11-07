"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        D = self.dim
        # Standard CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(D + 1))
        # Number of selected individuals
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2)**2 + mueff))
        
        # Initial mean and step size
        # Start at origin, sigma covers half the range initially? 
        # Box is [-5, 5], width 10. Sigma 2.0 is reasonable.
        x_mean = np.zeros(D)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(D)
        # Pre-compute square root of C for sampling
        # We will update C and recompute sqrt each generation
        # pc and ps are evolution paths
        pc = np.zeros(D)
        ps = np.zeros(D)
        
        # Initial population
        # We need to generate lam points per generation
        # Total budget 500. lam=11. ~45 generations.
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Track calls
        calls = 0
        
        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            return func(x)
        
        # Evaluate initial mean? 
        # Standard CMA-ES usually evaluates the population. 
        # We can start by evaluating the mean to get a baseline, 
        # or just start with the first population.
        # Let's evaluate the first population.
        
        while calls < self.budget and time.time() < deadline:
            # Generate population
            # Cholesky decomposition of C
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it
                C = np.eye(D)
                Csqrt = np.eye(D)
            
            # Sample lam points
            ys = np.array([np.random.standard_normal(D) for _ in range(lam)])
            xs = np.array([x_mean + sigma * (Csqrt @ y) for y in ys])
            
            # Clip to bounds? 
            # The problem states func(x) clips to box. 
            # It's better to let the optimizer work in unconstrained space 
            # and rely on the clipping, or use a penalty. 
            # However, BBOB instances are usually defined on R^D, 
            # but the evaluator clips. 
            # If we sample outside, the function value will be for the clipped point.
            # This is effectively a box constraint handled by the black box.
            
            # Evaluate population
            fs = []
            for x in xs:
                f = evaluate(x)
                if f is None:
                    break
                fs.append(f)
            
            if not fs:
                break
            
            fs = np.array(fs)
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
                
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Select mu best
            xs_best = xs_sorted[:mu]
            fs_best = fs_sorted[:mu]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_best, axis=0)
            
            # Update covariance matrix
            # Standard CMA-ES update
            # DinvCinv = np.linalg.inv(Csqrt) # Not needed directly if we use the standard formula
            
            # Evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Csqrt.T @ (x_mean_new - x_mean) / sigma)
            
            # hsig
            # Check if ps is too long
            # E[||ps||] = sqrt(D) * (1 - (1-cs)^lam)
            # This is an approximation. 
            # Simpler check: 
            norm_ps = np.linalg.norm(ps)
            # Theoretical norm
            # For now, use a simple heuristic or standard check
            # Let's use the standard hsig calculation
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^2*lam) < (1.4 + 2/(D+1)) * E||N(0,I)||
            # E||N(0,I)|| = sqrt(D) * (1 - 1/(4D) - 1/(21D^2))
            e_norm = np.sqrt(D) * (1 - 1/(4*D) - 1/(21*D**2))
            hsig = 1.0 if (norm_ps / np.sqrt(1 - (1-cs)**2 * lam)) < (1.4 + 2/(D+1)) * e_norm else 0.0
            
            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum wi * (xi - x_mean)(xi - x_mean).T
            delta_hs = (1 - hsig) * cc * (2 - cc)
            
            # Correction term
            # We need the outer products of the selected differences
            diffs = xs_best - x_mean
            # Weighted sum of outer products
            # W is weights
            C_update = (1 - c1 - cmu) * C
            C_update += c1 * (np.outer(pc, pc) + delta_hs * C)
            C_update += cmu * (diffs.T @ (weights[:, None] * diffs))
            
            # Symmetrize
            C = (C_update + C_update.T) / 2
            
            # Update sigma
            # chi = sqrt(D) * (1 - 1/(4D) - 1/(21D^2))
            chi = e_norm
            sigma = sigma * np.exp(cs / (D + 3) * (norm_ps / chi - 1))
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-8:
                break
            if sigma > 100:
                sigma = 100
                
            x_mean = x_mean_new
            
            # Check convergence
            if sigma < 1e-6:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
