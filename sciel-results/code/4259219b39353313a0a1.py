"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Global random sampling + CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global random sampling
        n_global = 50
        if budget <= n_global:
            # If budget is very small, just sample
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # Collect global samples
        xs = []
        fs = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            xs.append(x)
            fs.append(fx)
        
        xs = np.array(xs)
        fs = np.array(fs)
        
        # Sort by fitness
        order = np.argsort(fs)
        xs_sorted = xs[order]
        fs_sorted = fs[order]
        
        best_x_global = xs_sorted[0].copy()
        best_f_global = fs_sorted[0]

        # Phase 2: CMA-ES
        # Determine population size
        popsize = 4 + int(3 * np.log(dim + 1))
        popsize = min(popsize, max(2, (budget - n_global) // 2)) # Ensure enough budget for at least 2 gens?
        
        # Actually, we need enough budget for at least 1 generation of CMA-ES
        # Remaining budget
        rem_budget = budget - n_global
        
        if rem_budget < popsize:
            # Not enough for CMA-ES, do local random search from best global
            x = best_x_global.copy()
            fx = best_f_global
            for _ in range(rem_budget):
                y = x + 0.1 * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx:
                    fx = fy
                    x = y.copy()
            return x, fx

        # Initialize CMA-ES using the best 'popsize' points from global search
        # This provides a better initial distribution than identity covariance
        init_points = xs_sorted[:popsize]
        
        # Mean
        x = np.mean(init_points, axis=0)
        
        # Covariance
        # Center the points
        centered = init_points - x
        C = np.cov(centered.T)
        # Regularize to ensure positive definite
        if dim == 1:
            C = np.array([[max(C[0,0], 1e-4)]])
        else:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-4, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        # Sigma
        # Estimate sigma from the spread of the initial points
        sigma = np.mean(np.linalg.norm(centered, axis=1))
        sigma = np.clip(sigma, 0.1, 5.0)
        
        # CMA-ES parameters
        cs = (popsize + 2) / (dim + 4)
        ds = 1 + 2 * max(0, (np.sqrt((popsize + 1) / (dim + 3)) - 1) / (dim + 1)) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + popsize)
        cmu = min(1 - c1, 2 * (popsize - 2 + 1.0 / popsize) / ((dim + 2) ** 2 + popsize))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Weights
        w = np.exp(-5 * np.arange(popsize) / popsize)
        w /= w.sum()
        mu_eff = (w ** 2).sum() ** -1

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        global_best_x = best_x_global.copy()
        global_best_f = best_f_global

        calls_left = rem_budget

        while calls_left >= popsize:
            # Generate population
            z = rng.standard_normal((popsize, dim))
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            
            candidates = np.array([
                np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(popsize)
            ])
            
            # Evaluate
            fvals = np.array([func(c) for c in candidates])
            calls_left -= popsize
            
            # Update global best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < global_best_f:
                global_best_f = fvals[idx_best]
                global_best_x = candidates[idx_best].copy()
            
            # Sort
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            x_sorted = candidates[order]
            
            # Save old mean
            x_old = x.copy()
            
            # Update mean
            x = np.sum(w[:, None] * x_sorted, axis=0)
            
            # Update ps
            # y = (x_new - x_old) / sigma
            y_mean = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
            
            # Update pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
            
            # Update C
            delta_hs = (np.sum(ps ** 2) / (1 - (1 - cs) ** 2)) / (dim * (1 - (1 - cs) ** 2)) - 1
            hsig = 1.0 if delta_hs < (1 + 4 / (dim + 1)) else 0.0
            
            # Standard update
            C = (1 + c1 + cmu - c1 * hsig) * C
            C += c1 * (np.outer(pc, pc) - (1 - hsig) * C)
            
            # Rank-mu update
            # H = sum w_i (y_i - y_bar)(y_i - y_bar)^T
            # y_i = (x_i - x_old) / sigma
            # y_bar = (x - x_old) / sigma = y_mean
            # So H = (1/sigma^2) * sum w_i (x_i - x_old)(x_i - x_old)^T - ...?
            # Actually, the standard formula uses the sorted candidates relative to the old mean.
            # Let's compute the rank-mu update term carefully.
            # The term is cmu * sum_{i=1}^{mu} w_i * (z_i - z_bar)(z_i - z_bar)^T
            # where z_i are the step vectors.
            
            steps = (x_sorted - x_old) / sigma
            # We only use the top 'mu' steps? No, w is defined for all, but typically only top mu are non-zero or significant.
            # In our definition, w has non-zero values for all, but decays.
            # The standard CMA-ES uses mu = floor(lambda/2) or similar for the rank-mu update?
            # Actually, the weights w are used for both mean and rank-mu.
            # The formula is:
            # C += cmu * sum_{i=1}^{lambda} w_i (y_i - y_bar)(y_i - y_bar)^T
            # where y_i are the sorted steps.
            
            # Let's compute the outer products
            # Note: y_bar is the weighted mean of the steps?
            # x = sum w_i x_i
            # x_old + sigma * y_bar = x
            # y_bar = (x - x_old) / sigma
            # So y_bar is indeed the weighted mean of the steps y_i = (x_i - x_old)/sigma
            
            rank_mu_term = np.zeros((dim, dim))
            for i in range(popsize):
                diff = steps[i] - y_mean
                rank_mu_term += w[i] * np.outer(diff, diff)
                
            C += cmu * rank_mu_term
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-5, 10.0)

        # Use remaining calls
        while calls_left > 0:
            # Generate single point or small batch
            n_gen = min(popsize, calls_left)
            z = rng.standard_normal((n_gen, dim))
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            candidates = np.array([
                np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(n_gen)
            ])
            fvals = np.array([func(c) for c in candidates])
            calls_left -= n_gen
            
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < global_best_f:
                global_best_f = fvals[idx_best]
                global_best_x = candidates[idx_best].copy()
            
            # Update mean if we found something better?
            # For simplicity, just keep the best point found so far as the reference for final return.
            # But we can update x to the best point found in this last batch to continue if we had more budget.
            if fvals[idx_best] < global_best_f: # This check is redundant if we updated global_best_f
                pass
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
