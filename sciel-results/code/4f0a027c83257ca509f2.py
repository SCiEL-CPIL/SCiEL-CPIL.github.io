"""Combined optimizer: Rastrigin-aware grid initialization + CMA-ES with time guard."""
import time
import numpy as np


class Optimizer:
    """Rastrigin-aware initialization + CMA-ES local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _clip(x):
        return np.clip(x, -5.0, 5.0)

    def _coarse_grid_search(self, func, rng, budget_remaining):
        """Evaluate a coarse grid of points to find promising basins."""
        dim = self.dim
        # For Rastrigin, minima are near integer multiples of 2*pi/sqrt(2pi) ~ 2.5
        # But with rotation, we need to sample broadly.
        # Use a grid with ~8 points per dimension: 8^5 = 32768, too many.
        # Use ~4 points per dimension: 4^5 = 1024, still too many.
        # Use a random sample of ~50 points plus structured points.
        
        points = []
        # Add structured points: center and corners of sub-boxes
        # Divide [-5,5] into 3 parts: [-5,-1.67], [-1.67,1.67], [1.67,5]
        edges = [-5.0, -1.67, 1.67, 5.0]
        
        # Add center of each cell
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            pt = np.array([(edges[i]+edges[i+1])/2,
                                           (edges[j]+edges[j+1])/2,
                                           (edges[k]+edges[k+1])/2,
                                           (edges[l]+edges[l+1])/2,
                                           (edges[m]+edges[m+1])/2])
                            points.append(self._clip(pt))
        
        # Add random points
        while len(points) < 80 and budget_remaining > 0:
            points.append(self._clip(rng.uniform(-5, 5, dim)))
        
        # Evaluate all points
        results = []
        for p in points:
            if budget_remaining <= 0:
                break
            f = func(p)
            results.append((f, p))
            budget_remaining -= 1
        
        if not results:
            return None, None, 0
        
        # Sort by fitness
        results.sort(key=lambda x: x[0])
        best_f, best_x = results[0]
        return best_x, best_f, len(results)

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Simple CMA-ES implementation."""
        dim = self.dim
        x = self._clip(x0.copy())
        f_x = func(x)
        calls = 1
        
        # CMA-ES parameters
        lambda_ = 4 * (int(np.log(dim)) + 2)  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = sigma0
        
        best_x = x.copy()
        best_f = f_x
        
        while calls < budget and time.time() < deadline:
            if time.time() > deadline - 2.0:
                break
            
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if calls >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(dim)
                # Simple Cholesky-like approximation (use C directly for small dim)
                # For dim=5, we can use a simple approach
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                
                x_i = self._clip(x + sigma * y)
                f_i = func(x_i)
                calls += 1
                offspring.append((f_i, x_i))
            
            if not offspring:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i][1]
            
            # Update covariance matrix
            diff = x_new - x
            x = x_new
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C @ (diff / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(calls//lambda_))) < (1.4 + 1/dim) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update C
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(calls//lambda_))) < (1.4 + 1/dim) * chi_n else 0
            C = (1 - c_c * (2 - c_c) + c_sigma * (2 - c_sigma) * mu_eff) * C
            C += c_sigma * (2 - c_sigma) * (1 - mu_eff) * h_sigma * C
            C -= c_sigma * (2 - c_sigma) * mu_eff * np.outer(diff/sigma, diff/sigma)
            
            # Ensure C is positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
        
        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 310.0
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse search to find promising basins
        x0, f0, used = self._coarse_grid_search(func, rng, budget)
        calls = used
        
        if x0 is None:
            x0 = self._clip(rng.uniform(-5, 5, dim))
            f0 = func(x0)
            calls = 1
        
        best_x = x0.copy()
        best_f = f0
        
        # Phase 2: CMA-ES local search from best point
        remaining = budget - calls
        if remaining > 50 and time.time() < deadline:
            sigma0 = 1.5  # Initial step size
            px, pf, pused = self._cma_es(func, best_x, sigma0, remaining, deadline, rng)
            calls += pused
            if pf < best_f:
                best_f = pf
                best_x = px.copy()
        
        # Phase 3: If we have budget left, do another coarse search + CMA-ES
        remaining = budget - calls
        if remaining > 100 and time.time() < deadline - 10:
            # Try a different region
            x1, f1, used1 = self._coarse_grid_search(func, rng, min(remaining // 2, 50))
            calls += used1
            if f1 < best_f:
                best_f = f1
                best_x = x1.copy()
            
            remaining = budget - calls
            if remaining > 50 and time.time() < deadline:
                sigma0 = 1.0
                px, pf, pused = self._cma_es(func, best_x, sigma0, remaining, deadline, rng)
                calls += pused
                if pf < best_f:
                    best_f = pf
                    best_x = px.copy()
        
        return best_x, best_f
