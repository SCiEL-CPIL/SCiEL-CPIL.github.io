"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer for BBOB Sphere (f1).
    Uses CMA-ES with population size 10 to model full covariance structure,
    automatically aligning with hidden rotations and adapting step sizes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:  # Safety margin
            return None
        self._calls_made += 1
        try:
            val = func(x)
        except Exception:
            val = 1e10
        if np.isnan(val) or np.isinf(val):
            val = 1e10
        val = float(val)
        if val < self._best_f:
            self._best_f = val
            self._best_x = np.array(x, dtype=float).copy()
        return val

    def _cma_es(self, func, x0, sigma0, max_calls):
        """
        CMA-ES implementation.
        """
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # ~10 for n=5
        pop_size = min(pop_size, max_calls - 5)
        if pop_size < 4:
            pop_size = max(4, max_calls // 2)
        
        # CMA-ES parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # State
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = np.sqrt((n + 2) * (c_sigma + 1 - c_sigma))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Evaluate initial point
        fx = self._evaluate(func, x)
        if fx is None:
            return x, self._best_f
        
        rng = np.random.default_rng(42)
        
        while self._calls_made < self.budget - 2:
            if time.time() - self._start_time > 290:
                break
            
            # Generate population
            samples = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                # Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                samples.append(xi)
            
            # Evaluate population
            fvals = []
            for xi in samples:
                fxi = self._evaluate(func, xi)
                if fxi is None:
                    fxi = 1e10
                fvals.append(fxi)
            
            fvals = np.array(fvals)
            samples = np.array(samples)
            
            # Sort by fitness
            idx = np.argsort(fvals)
            samples = samples[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < self._best_f:
                self._best_f = fvals[0]
                self._best_x = samples[0].copy()
            
            # Weighted mean
            new_x = np.zeros(n)
            for i in range(mu):
                new_x += weights[i] * samples[i]
            
            # Update step size
            diff = new_x - x
            ps = (1 - c_sigma) * ps + cs * np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * diff
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)
            
            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff / sigma
            
            # Update covariance matrix
            delta_diag = (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1) * c_sigma * (2 - c_sigma)
            C = (1 + c_1 + c_mu - c_sigma) * C \
                + c_1 * (2 - c_c) * np.outer(pc, pc) \
                + c_mu * np.sum(weights[:mu, None] * np.outer(samples[:mu] - x, samples[:mu] - x) / (sigma ** 2), axis=0) \
                + delta_diag * C
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            x = new_x
            
            # Check convergence
            if np.max(fvals[:mu]) - np.min(fvals[:mu]) < 1e-12:
                break
            if np.linalg.norm(diff) < 1e-12 * (1 + np.linalg.norm(x)):
                break
        
        return x, self._best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf

        # Phase 1: Fast initial probe for early AOCC
        center = np.zeros(dim)
        fx_center = self._evaluate(func, center)
        
        # Evaluate a few axis points to get a better initial estimate
        for i in range(dim):
            if self._calls_made >= budget or time.time() - self._start_time > 290:
                break
            for sign in [-1, 1]:
                x = center.copy()
                x[i] = sign * 2.0
                self._evaluate(func, x)
        
        # Phase 2: CMA-ES for the remaining budget
        remaining = budget - self._calls_made
        if remaining > 10:
            x0 = self._best_x if self._best_x is not None else center
            x_final, f_final = self._cma_es(func, x0, 1.5, remaining)
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
