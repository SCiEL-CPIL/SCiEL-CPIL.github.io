"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive population size for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        CMA-ES implementation.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        c_sigma = (lam + 2) / ((dim + 1.3)**2 + lam)
        c_c = (4 + lam / dim) / ((dim + 1.3)**2 + 4 + lam / dim)
        c_1 = 2 / ((dim + 1.43)**2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) * (mu_eff + 5)) - 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1) - 1) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Weighted mean of top mu individuals
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        # Approximate inverse square root using Cholesky
        # For simplicity, we'll use a simpler update strategy
        
        best_x = x.copy()
        best_f = fx
        
        while calls < max_calls:
            if time.time() > deadline - 0.5:
                break
            
            # Generate population
            ys = []
            xs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Simple approximation: use current C as diagonal
                # For better performance, we could do proper Cholesky
                y = sigma * z  # Simplified: no rotation yet
                xi = x + y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if calls >= max_calls:
                    break
                ys.append(z)
                xs.append((xi, fi))
            
            if len(xs) < 2:
                break
            
            # Sort by fitness
            xs.sort(key=lambda t: t[1])
            
            # Update best
            if xs[0][1] < best_f:
                best_f = xs[0][1]
                best_x = xs[0][0].copy()
            
            # Weighted mean
            m = np.zeros(dim)
            for i in range(min(mu, len(xs))):
                m += weights[i] * xs[i][0]
            
            # Update sigma
            # p_sigma update
            if np.linalg.norm(C_sqrt_inv @ (m - x)) > 1.4 * (1 + c_sigma) * chi_n:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (m - x) / sigma
            
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / (1 - c_sigma**lam) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            
            # C update (simplified)
            # For proper CMA-ES, we'd do full matrix updates
            # Here we use a simplified rank-1 update
            delta_x = (m - x) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(delta_x, delta_x))
            for i in range(min(mu, len(xs))):
                y_i = xs[i][0] - x
                C += c_mu * weights[i] * np.outer(y_i / sigma, y_i / sigma)
            
            # Renormalize C
            C = (C + C.T) / 2
            C /= np.trace(C) / dim
            
            x = m
            
            if time.time() > deadline - 0.5:
                break
        
        return best_x, best_f, calls

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        c_inc = 1.1
        c_dec = 0.9
        
        best_x = x.copy()
        best_f = fx
        
        while calls < max_calls:
            if time.time() > deadline - 0.5:
                break
            
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= c_inc
            else:
                sigma *= c_dec
            
            sigma = np.clip(sigma, 1e-8, 10.0)
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Strategy: Multi-start with CMA-ES for better handling of ill-conditioning
        # Phase 1: CMA-ES from center
        remaining = budget
        max_calls_p1 = min(100, remaining)
        
        if max_calls_p1 > 5:
            bx, bf, used = self._cma_es(func, rng, np.zeros(dim), 2.0, max_calls_p1, deadline)
            total_calls += used
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # Phase 2: Multiple (1+1)-ES restarts from diverse points
        num_restarts = 6
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5.0:
            calls_per_restart = max(10, remaining // (num_restarts + 1))
            
            starts = [best_x.copy()]
            for _ in range(num_restarts - 1):
                x0 = rng.uniform(-4.5, 4.5, size=dim)
                starts.append(x0)
            
            for x0 in starts:
                if total_calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                rem = budget - total_calls
                max_calls = min(calls_per_restart, rem)
                if max_calls <= 1:
                    break
                
                sigma0 = 1.0 if np.array_equal(x0, best_x) else 2.0
                bx, bf, used = self._es_1plus1(func, rng, x0, sigma0, max_calls, deadline)
                total_calls += used
                
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()
        
        # Phase 3: Final exploitation with CMA-ES from best point
        remaining = budget - total_calls
        if remaining > 20 and time.time() < deadline - 5.0:
            bx, bf, used = self._cma_es(func, rng, best_x, 0.5, remaining, deadline)
            total_calls += used
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # Phase 4: Final fine-tuning with (1+1)-ES
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline - 2.0:
            bx, bf, used = self._es_1plus1(func, rng, best_x, 0.1, remaining, deadline)
            total_calls += used
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
