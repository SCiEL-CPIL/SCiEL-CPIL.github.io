"""Optimizer for BBOB f6 (Attractive Sector) combining CMA-ES with global scan."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with a global scan warm start and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline):
        """Small CMA-ES. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        rng = np.random.default_rng()
        x = np.array(x0, dtype=float)
        fx = float(func(x))
        calls = 1
        sigma = float(sigma0)
        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Parameters for CMA-ES
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4.0 + mueff / n) / ((n + 2.0) ** 2 + mueff / 2.0)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (2.0 - cc + cs)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) /
                 ((n + 2.0) ** 2 + mueff))
        chi = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        
        # We use a subset of the population for the update to save calls if needed, 
        # but standard CMA-ES uses mu = n//2. Let's stick to standard.
        
        while calls < max_calls and time.time() < deadline:
            # Generate population
            batch_y = []
            batch_z = []
            batch_f = []
            
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x + sigma * z
                batch_y.append(y)
                batch_z.append(z)
                fy = float(func(y))
                batch_f.append(fy)
                calls += 1
                if calls >= max_calls or time.time() > deadline:
                    break
            
            if len(batch_y) < 2:
                break

            batch_y = np.array(batch_y)
            batch_z = np.array(batch_z)
            batch_f = np.array(batch_f)
            
            # Sort by fitness
            order = np.argsort(batch_f)
            batch_y = batch_y[order]
            batch_z = batch_z[order]
            batch_f = batch_f[order]
            
            # Update best
            if batch_f[0] < fx:
                fx = batch_f[0]
                # We don't necessarily need to update x to the best immediately for CMA-ES mean, 
                # but it's good for tracking.
                
            # Weighted mean
            xmean = np.sum(w[:, None] * batch_y, axis=0)
            
            # Update ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (Cinv @ (batch_z.mean(axis=0) * np.sqrt(mueff)))
            
            # Check for delta sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls + 1)))) < (1.4 + 2.0 / (n + 1)) * chi
            
            # Update pc
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc)) * (xmean - x) / sigma
            
            # Update C
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) \
                + cmu * np.sum(w[:, None, None] * np.outer(batch_y - x, batch_y - x), axis=0) / (sigma ** 2 + 1e-300)
            
            # Recompute inverse (n=5 is small)
            try:
                Cinv = np.linalg.inv(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                Cinv = np.eye(n)
                C = np.eye(n)
            
            # Update sigma
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1.0))
            
            # Update mean
            x = xmean
            
            # Convergence check
            if sigma < 1e-12 or not np.isfinite(sigma):
                break
                
        return x, fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 290.0  # Safety margin

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            calls_used += 1
            return float(func(np.clip(np.asarray(x, dtype=float), -5.0, 5.0)))

        # Phase 1: Global Scan to find a good starting point
        # f6 has an attractive sector, so random points might fall into it.
        n_scan = min(budget // 5, 50) # Small scan
        rng = np.random.default_rng(42)
        
        for _ in range(n_scan):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            if calls_used >= budget or time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Phase 2: CMA-ES from the best point
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            # Start with a moderate sigma
            sigma0 = 1.0 
            try:
                x, fx, used = self._cmaes(safe_func, best_x, sigma0, remaining, deadline)
                calls_used += used
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            except Exception:
                pass

        # Phase 3: If we have time and budget, maybe a second run from a slightly perturbed best?
        # Or just return. For 500 budget, one CMA-ES run is usually sufficient.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
