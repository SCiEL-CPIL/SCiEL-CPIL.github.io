import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Random sampling to find a promising region
        n_global = 50
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: CMA-ES with adaptive population size
        remaining = self.budget - self.call_count
        if remaining > 30:
            try:
                self._cmaes_adaptive(wrapped_func, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Fine-grained local polish
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._fine_polish(wrapped_func, self.best_x.copy(), min(remaining, 40))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _cmaes_adaptive(self, func, x_start, max_calls):
        """CMA-ES with adaptive population size and robust stopping."""
        n = self.dim
        x = x_start.copy()
        sigma = 1.0
        
        # Start with larger population for better exploration
        lam = 30
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial state
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_cholesky = np.eye(n)
        
        calls_used = 0
        best_f_so_far = self.best_f
        stagnation_counter = 0
        no_improve = 0
        
        for iteration in range(300):
            if calls_used >= max_calls or time.time() > self.deadline:
                break
            
            # Adapt population size: reduce if stagnating
            if stagnation_counter > 3 and lam > 10:
                lam = max(10, lam - 2)
                mu = lam // 2
                weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)
                c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
                c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Sample candidates
            candidates = []
            for j in range(lam):
                if calls_used + (lam - j) > max_calls or time.time() > self.deadline:
                    break
                z = np.random.randn(n)
                x_candidate = x + sigma * (C_cholesky @ z)
                x_candidate = np.clip(x_candidate, -self.bound, self.bound)
                candidates.append(x_candidate)
                calls_used += 1
            
            if len(candidates) < mu:
                break
            
            # Evaluate
            fvals = []
            for xc in candidates:
                if calls_used >= max_calls or time.time() > self.deadline:
                    break
                fvals.append(func(xc))
            
            if len(fvals) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals[:len(candidates)])
            sorted_x = np.array([candidates[i] for i in indices])
            sorted_f = np.array([fvals[i] for i in indices])
            
            # Update mean with weighted recombination
            w = weights[:mu]
            x_new = sum(w[i] * sorted_x[i] for i in range(mu))
            
            # Update p_sigma
            y_sigma = (x_new - x) / sigma if sigma > 1e-30 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_sigma
            
            # Update covariance
            rank_one = np.outer(p_c, p_c)
            y_vecs = np.array([(sorted_x[i] - x) / (sigma + 1e-30) for i in range(mu)])
            rank_mu = np.sum([w[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize and factorize
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C + 1e-20 * np.eye(n))
                C_cholesky = L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C_cholesky = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = x_new
            
            # Track improvement
            if self.best_f < best_f_so_far - 1e-8:
                best_f_so_far = self.best_f
                stagnation_counter = 0
                no_improve = 0
            else:
                stagnation_counter += 1
                no_improve += 1
            
            # Early stopping criteria
            if sigma < 1e-10:
                break
            if no_improve > 15:
                break
            if np.linalg.norm(x_new - x_start) < 1e-10 and np.linalg.norm(y_sigma) < 1e-8:
                break

    def _fine_polish(self, func, x_start, max_calls):
        """Very fine local search with tiny step sizes."""
        n = self.dim
        x = x_start.copy()
        f_best = func(x)
        calls = 1
        step = 0.01
        for it in range(max_calls - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            improved = False
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-12:
                        x = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
