"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES-like optimizer with coarse grid initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        # Phase 1: Coarse Grid Search to find a good starting point
        # Bent Cigar is narrow. A grid helps locate the valley.
        # 5 dimensions, let's try a 3x3x3x3x3 grid = 243 points.
        # Or 4x4x4x4x4 = 1024 points. 
        # Budget is 500. 
        # Let's use a stratified random or simple grid.
        # Simple grid: 3 points per dim -> 243 calls. Leaves ~250 for CMA-ES.
        
        grid_size = 3
        points = []
        for i1 in range(grid_size):
            for i2 in range(grid_size):
                for i3 in range(grid_size):
                    for i4 in range(grid_size):
                        for i5 in range(grid_size):
                            x = np.array([
                                -5 + 10 * (i1 / (grid_size - 1)),
                                -5 + 10 * (i2 / (grid_size - 1)),
                                -5 + 10 * (i3 / (grid_size - 1)),
                                -5 + 10 * (i4 / (grid_size - 1)),
                                -5 + 10 * (i5 / (grid_size - 1))
                            ])
                            points.append(x)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate grid
        for x in points:
            if time.time() > deadline - 10: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
        
        # If we ran out of time or budget, return best found
        calls_made = len(points)
        if best_x is None:
             return np.zeros(self.dim), np.inf
             
        if time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: CMA-ES-like refinement around best_x
        # We have remaining budget.
        # CMA-ES parameters:
        # lambda = 4 * (4 + floor(log(dim)))
        lam = 4 * (4 + int(np.log(self.dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = (2 / (self.dim**1.8 + 10)) * (1 - cc - cs)
        cmu = min(1 - c1, 2 * ((mueff + 2) / (self.dim + 2) - 1) * (1 - c1))
        
        x_mean = best_x.copy()
        sigma = 1.0 # Start with moderate sigma
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # We need to track calls. Grid used ~243.
        # Remaining budget: 500 - 243 = 257.
        # CMA-ES needs lambda calls per iteration.
        # 257 / 46 (approx lam) ~ 5-6 iterations.
        # This might be too few for CMA-ES to converge well if sigma is wrong.
        
        # Alternative: Use a simpler (1+1)-ES but with better initial sigma and maybe a few restarts?
        # Or a simpler adaptive step size search.
        
        # Let's stick to a robust (1+1)-ES variant but with the benefit of the good start.
        # Actually, CMA-ES is better for rotated cigar. But 5 iterations is short.
        # Let's try a smaller lambda for CMA-ES to get more iterations?
        # lambda = 4 * (4 + log(5)) ~ 4*5 = 20? No, standard is higher.
        # Let's use lambda = 10.
        lam = 10
        mu = 5
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = (2 / (self.dim**1.8 + 10)) * (1 - cc - cs)
        cmu = min(1 - c1, 2 * ((mueff + 2) / (self.dim + 2) - 1) * (1 - c1))
        
        x_mean = best_x.copy()
        sigma = 0.5 # Start smaller, we are likely close
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        remaining_budget = self.budget - calls_made
        
        while remaining_budget >= lam and time.time() < deadline - 5:
            # Generate offspring
            ys = []
            for _ in range(lam):
                # Sample from N(0, C)
                # Use Cholesky or eigen decomposition
                # For dim=5, eigen is fine
                vals, vecs = np.linalg.eigh(C)
                # Ensure positive
                vals = np.maximum(vals, 1e-10)
                z = np.random.randn(self.dim)
                y = vecs @ (np.sqrt(vals) * z)
                y = x_mean + sigma * y
                y = np.clip(y, -5, 5)
                ys.append(y)
                
            # Evaluate
            fvals = []
            for y in ys:
                if time.time() > deadline - 2: break
                fvals.append(func(y))
            
            if len(fvals) < lam:
                break
                
            remaining_budget -= lam
            
            # Sort
            order = np.argsort(fvals)
            ys = [ys[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = ys[0]
                
            # Update mean
            y_mean = np.average(ys[:mu], axis=0, weights=weights)
            y_diff = y_mean - x_mean
            x_mean = y_mean
            
            # Update step size
            cs = (mueff + 2) / (self.dim + mueff + 5)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C ** -0.5) @ y_diff / sigma
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (self.dim + 1)) * np.trace(C) / self.dim else 0
            
            sigma *= np.exp(cs / self.dim * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update covariance
            pc = (1 - cc) * pc + hsig * cc * y_diff / sigma
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C) + cmu / mueff * sum(weights[i] * np.outer(ys[i] - x_mean, ys[i] - x_mean) for i in range(mu))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            vals, vecs = np.linalg.eigh(C)
            if np.any(vals < 1e-10):
                C = np.eye(self.dim)
                sigma = 1.0
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
