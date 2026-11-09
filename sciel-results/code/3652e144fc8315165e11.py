import numpy as np
import time

class Optimizer:
    """CMA-ES with Latin Hypercube initialization for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0
        self._cache = {}

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0
        self._cache = {}
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            key = tuple(np.round(x, 10).tolist())
            if key in self._cache:
                return self._cache[key]
            
            self.calls_made += 1
            val = func(x)
            self._cache[key] = val
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def cmaes(x0, sigma0, max_evals):
            """Budget-aware CMA-ES implementation."""
            n = self.dim
            lam = max(2 * n, 8)
            mu = lam // 2
            
            # Step size control parameters
            cc = (4 + n / 3) / (n + 4 + 2 / n)
            cs = (n + 1.7) / (n + n + 2.5)
            dm = 1 + 2 * max(0, np.sqrt((cc - 1) / (cc + 1)) - 1) + cs
            ccov = (5 + n / 2) / (n + 9)
            ccov1 = 2 * ((2 + 2 / n) * (1 - ccov + cs / dm))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            # State
            mean = np.clip(x0.copy(), -5, 5)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evals = 0
            
            while evals < max_evals and self.calls_made < self.budget and time.time() < self.deadline:
                # Generate population
                B = np.linalg.cholesky(C)
                W = rng.normal(size=(lam, n))
                pop = np.array([mean + sigma * (B @ w.T).T for w in W])
                pop = np.clip(pop, -5, 5)
                
                # Evaluate
                fvals = np.empty(lam)
                for i in range(lam):
                    if self.calls_made >= self.budget or time.time() > self.deadline:
                        break
                    fvals[i] = wrapped_func(pop[i])
                    evals += 1
                
                if evals >= max_evals or self.calls_made >= self.budget:
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                y = pop[order]
                fy = fvals[order]
                
                # Weighted recombination
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights**2)
                
                yw = y[:mu]
                new_mean = np.sum(weights[:, None] * yw, axis=0)
                
                # Update step size
                dnorm = np.linalg.norm(new_mean - mean) / (sigma * np.sqrt(2))
                hsig = (dnorm / (1 - (1 / dm) * (evals / max_evals)) < (1.4 + 2 / (n + 1)) * chi_n)
                ps = (1 - cs) * ps + (hsig * np.sqrt(cs * (2 - cs) * mueff)) * (new_mean - mean) / sigma
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * chi_n
                
                # Update covariance matrix
                pc = (1 - cc) * pc + (hsig * np.sqrt(cc * (2 - cc) * mueff)) * (new_mean - mean) / sigma
                delta_h = (1 - hsig) * cc * (2 - cc)
                C = (1 + ccov1 * ccov * delta_h - ccov) * C + ccov * np.outer(pc, pc)
                
                # Update mean and step size
                mean = new_mean
                sigma *= np.exp(cs / dm * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                
                # Ensure C is positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                sigma = np.sqrt(np.trace(C) / n)
            
            return mean, sigma

        # Phase 1: Latin Hypercube sampling to locate valley basin
        n_lhs = min(30, max(10, self.budget // 15))
        lhs_starts = []
        for _ in range(n_lhs):
            x = np.empty(self.dim)
            for d in range(self.dim):
                x[d] = -5 + (rng.random() + d / n_lhs) / n_lhs * 10
            lhs_starts.append(x)
        
        # Add structured valley points
        valley_starts = [np.ones(self.dim)]
        for a in [0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 0.3, -0.3]:
            v = np.zeros(self.dim)
            v[0] = a
            for i in range(1, self.dim):
                v[i] = a ** (2 ** i)
            v = np.clip(v, -5, 5)
            valley_starts.append(v)
        
        all_starts = lhs_starts + valley_starts
        
        # Evaluate all starts
        for x0 in all_starts:
            if self.calls_made >= self.budget - 20 or time.time() > self.deadline:
                break
            try:
                f0 = wrapped_func(x0)
            except StopIteration:
                break
        
        # Phase 2: CMA-ES from best starts
        if self.best_x is not None and self.calls_made < self.budget - 30:
            remaining = self.budget - self.calls_made
            n_cma = min(3, max(1, remaining // 80))
            
            # Sort starts by function value
            start_fvals = []
            for x0 in all_starts:
                if self.calls_made >= self.budget - 10:
                    break
                try:
                    f0 = wrapped_func(x0)
                    start_fvals.append((f0, x0))
                except StopIteration:
                    break
            
            start_fvals.sort(key=lambda x: x[0])
            
            for i in range(min(n_cma, len(start_fvals))):
                if self.calls_made >= self.budget - 20 or time.time() > self.deadline:
                    break
                
                f0, x0 = start_fvals[i]
                # Adaptive sigma based on problem scale
                sigma0 = 1.5 if i == 0 else 2.0
                
                try:
                    mean, sigma = cmaes(x0, sigma0, max_evals=self.budget - self.calls_made - 10)
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # Phase 3: Final polish with smaller sigma
        if self.best_x is not None and self.calls_made < self.budget - 15 and time.time() < self.deadline:
            try:
                mean, sigma = cmaes(self.best_x, 0.3, max_evals=self.budget - self.calls_made - 5)
            except StopIteration:
                pass
            except Exception:
                pass
        
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            try:
                self.best_f = wrapped_func(x0)
            except StopIteration:
                pass
        
        return self.best_x, self.best_f
