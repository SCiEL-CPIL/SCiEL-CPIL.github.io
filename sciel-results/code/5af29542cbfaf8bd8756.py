"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts, adapted for the BBOB f19 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial broad exploration with a few random starts to find a good basin
        # We use a small population for CMA-ES.
        # lambda = 4 + floor(3 * log(dim)) is standard, but for dim=5, lambda=11 is typical.
        # Let's use lambda=10 for efficiency.
        lam = 10
        mu = lam // 2
        
        # We will run CMA-ES in a loop, restarting if it stagnates or if we run out of time.
        # Each restart uses a smaller budget slice or just runs until termination criteria.
        
        # Initial point: random
        x0 = np.random.uniform(-4, 4, self.dim)
        f0 = eval_func(x0)
        
        # If initial is bad, try a couple more
        for _ in range(2):
            x_try = np.random.uniform(-4, 4, self.dim)
            f_try = eval_func(x_try)
            if f_try < f0:
                x0, f0 = x_try, f_try

        # CMA-ES parameters
        sigma = 1.0
        C = np.eye(self.dim)
        # Eigenvalues of C are 1
        # Update rules
        # c_sigma = (mu_eff + 2) / ((n + 2) + mu_eff)
        # c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        # c_1 = 2 / ((n + 1.3)^2 + mu)
        # c_mu = (mu_eff / 2 - 1) / ((n + 2)^2 + mu_eff)
        
        n = self.dim
        mu_eff = 1.0 / sum(np.log(np.arange(lam) - np.arange(lam - mu) + 0.5)) # Approximation, better to calculate properly
        
        # Proper weights
        weights = np.log(mu + 0.5) - np.log(np.arange(lam) - np.arange(lam - mu) + 0.5)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / ((n + 2) + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = (mu_eff / 2 - 1) / ((n + 2)**2 + mu_eff)
        
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / n - 1) + 1) + c_sigma
        
        # Cholesky of C (starts as I)
        # We need to update C and maintain its Cholesky factor or use eigendecomposition periodically.
        # For simplicity and speed in Python, we might use a simplified update or just rely on the random steps being good enough?
        # No, CMA-ES is the point. But implementing full CMA-ES in Python without Cython might be slow?
        # Dim is 5, so matrix operations are cheap.
        
        # Let's implement a lightweight CMA-ES step.
        
        def cma_es_step(x_mean, sigma, C, pc, ps, weights, c_sigma, c_c, c_1, c_mu, d_sigma, lam, mu, n):
            # Generate population
            # We need eigendecomposition of C to generate samples
            # For n=5, np.linalg.eigh is fast.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Sample z_i ~ N(0, I)
            Z = np.random.randn(lam, n)
            # Y = B * D * Z
            # B = eigvecs, D = sqrt(eigvals)
            Y = Z @ (eigvecs * np.sqrt(eigvals)).T
            
            # X = x_mean + sigma * Y
            X = x_mean + sigma * Y
            
            # Evaluate
            F = np.array([eval_func(x) for x in X])
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            x_mean_new = sum(weights[i] * X[i] for i in range(mu))
            
            # Update step size control vector ps
            # Need B^-1 * (x_mean_new - x_mean) / sigma
            # B = eigvecs, D = sqrt(eigvals)
            # B^-1 = eigvecs.T
            # D^-1 = 1/sqrt(eigvals)
            # (x_mean_new - x_mean) / sigma
            delta = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvecs.T @ (delta / np.sqrt(eigvals)))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (eigvecs.T @ (delta / np.sqrt(eigvals)))
            
            # Update C
            # h_sig = (||ps|| / sqrt(1 - (1-c_sigma)^{2*(gen+1)})) / (sqrt(n) * (1 + c_sigma))
            # We don't track gen easily here, approximate with 1
            h_sig = 1.0 # Simplified
            # Actually, let's just do the rank-1 and rank-mu update
            
            # Rank 1
            # C = (1 - c_1) * C + c_1 * (pc * pc.T + (1 - (1-c_c)^2) * (h_sig ? 1 : 0) * I)
            # Standard: C = (1 - c_1) * C + c_1 * (pc * pc.T + (1 - (1-c_c)**2) * (1 if h_sig else 0) * I)
            # Let's use the standard update without h_sig correction for simplicity, it's usually robust.
            # Actually, h_sig prevents divergence. Let's estimate it.
            # ||ps|| / sqrt(1 - (1-c_sigma)^(2*gen))
            # Since we don't have gen, let's just use 1.
            
            # Better: use the standard formula
            # h_sig = (||ps|| / sqrt(1 - (1-c_sigma)^(2*gen))) / (sqrt(n) * (1 + c_sigma))
            # Assuming gen is large enough, (1-c_sigma)^(2*gen) is small.
            # Let's just use h_sig = 1.0 for now, it's a heuristic.
            
            C = (1 - c_1) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(n))
            
            # Rank mu
            # Y_sorted = X[:mu]
            # Delta = (x_mean_new - x_mean) / sigma
            # Wait, the formula uses the selected points relative to the mean.
            # C += c_mu * sum(weights[i] * (X[i] - x_mean) @ (X[i] - x_mean).T) / sigma^2 ?
            # No, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + ...) + c_mu * sum(weights[i] * (Y[i] - y_mean) @ (Y[i] - y_mean).T)
            # Where Y are the standardized points.
            
            # Let's stick to a simpler, more robust variant:
            # C = (1 - c_1) * C + c_1 * (pc * pc.T)
            # This is "Rank-1 CMA-ES". It's slower but very stable and easy to implement.
            # Given the small budget and dimension, Rank-1 might be sufficient and faster to code correctly.
            
            # Let's implement Rank-1 CMA-ES.
            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, _ = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C += 1e-12 * np.eye(n)
                
            return x_mean_new, sigma, C, pc, ps

        # Main loop
        while calls < self.budget and time.time() < deadline:
            # Check for stagnation
            # If sigma becomes too small or too large, restart
            if sigma < 1e-6 or sigma > 10.0:
                break
            
            # Run a few steps of CMA-ES
            # To avoid infinite loops in a single step if it fails, we just do one step.
            # But CMA-ES needs multiple steps to converge.
            # Let's do a mini-run of CMA-ES with a sub-budget.
            
            sub_budget = min(50, self.budget - calls)
            if sub_budget <= 0:
                break
                
            x_mean = best_x.copy()
            # Reset C and paths for a fresh start? Or continue?
            # Restarting is safer to escape local minima.
            # But we have a good best_x. Let's continue from there with fresh C?
            # No, CMA-ES state is coupled. Let's restart the CMA-ES process centered at best_x.
            
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            sigma = 0.5 # Start with a smaller sigma for local search
            
            for _ in range(sub_budget // lam + 1):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                    
                x_mean, sigma, C, pc, ps = cma_es_step(x_mean, sigma, C, pc, ps, weights, c_sigma, c_c, c_1, c_mu, d_sigma, lam, mu, n)
                
                # Update global best
                # cma_es_step already updated best_x/best_f via eval_func
                
                # Check for stagnation
                if sigma < 1e-5:
                    break
                    
            # If we exhausted the sub-budget or stopped due to stagnation, check if we should restart with a wider search
            # If best_f hasn't improved significantly, we might be stuck.
            # For now, just continue the main loop which will re-initialize CMA-ES centered at the new best_x.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
