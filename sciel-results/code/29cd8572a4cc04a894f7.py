import numpy as np
import time
import scipy.stats.qmc as qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Sobol Sequence Exploration
        # Use 100 points for global coverage
        n_explore = min(100, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n_explore) * 10.0 - 5.0
        
        for x in sobol_points:
            if calls >= budget or time.time() > deadline:
                break
            try:
                safe_func(x)
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                safe_func(best_x)
            except Exception:
                pass

        # Phase 2: CMA-ES Exploitation
        if calls < budget and time.time() < deadline:
            try:
                self._run_cma_es(func, safe_func, best_x, calls, budget, deadline)
            except Exception:
                pass

        return best_x, best_f

    def _run_cma_es(self, func, safe_func, x0, calls, budget, deadline):
        # Simplified CMA-ES
        dim = self.dim
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Parameters
        c_sigma = (mu_eff / 2 + 1) / (dim + mu_eff / 2 + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))
        
        # State
        m = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1 / c_sigma
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Precompute
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        
        while calls < budget and time.time() < deadline:
            # Generate samples
            z = np.random.randn(lambda_pop, dim)
            x_samples = m + sigma * (z @ (B / D).T)
            
            # Evaluate
            f_values = []
            for x in x_samples:
                try:
                    f_values.append(safe_func(x))
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    f_values.append(np.inf)
            
            if len(f_values) < lambda_pop:
                break
                
            # Sort
            idx = np.argsort(f_values)
            f_values = np.array(f_values)[idx]
            x_samples = x_samples[idx]
            
            # Update Mean
            y_w = np.sum(weights[:, np.newaxis] * x_samples[:mu], axis=0)
            m = m + c_sigma * y_w # Wait, standard update is m = m + (1-c_sigma)*y_w? No.
            # Standard CMA-ES:
            # y_w = sum(w_i * (x_i - m_old) / sigma)
            # m_new = m_old + sigma * y_w
            # Let's re-calculate properly.
            
            y_w = np.sum(weights[:, np.newaxis] * (x_samples[:mu] - m) / sigma, axis=0)
            m_new = m + sigma * y_w
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B / D) @ y_w
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_pop))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            pc = (1 - c_1) * pc + h_sigma * np.sqrt(c_1 * (2 - c_1) * mu_eff) * y_w
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_1 * C) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * (y_w[:, np.newaxis] * y_w[np.newaxis, :]), axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                D, B = np.linalg.eigh(C)
                if np.min(D) < 1e-10:
                    D = np.maximum(D, 1e-10)
                D = np.sqrt(D)
            except:
                break
                
            m = m_new
            if time.time() > deadline:
                break
