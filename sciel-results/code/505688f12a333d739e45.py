import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, budget, dim, sigma_init=1.0, pop_size=None, max_generations=None):
        """Small-population CMA-ES for global search."""
        if pop_size is None:
            pop_size = 4
        if max_generations is None:
            max_generations = budget // (pop_size + 1)
        
        x = np.array(x0, dtype=float)
        C = np.eye(dim)
        sigma = sigma_init
        best_x = x.copy()
        best_f = np.inf
        calls = 0
        
        # CMA-ES parameters
        mu = pop_size // 2  # number of parents
        weights = np.zeros(mu)
        w_sum = 0.0
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum += weights[i]
        weights /= w_sum
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        for gen in range(max_generations):
            if calls >= budget:
                break
            
            # Generate population
            ys = []
            fs = []
            for i in range(pop_size):
                z = np.random.standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                    calls += 1
                    ys.append(y)
                    fs.append(fy)
                except Exception:
                    break
            
            if len(fs) < pop_size:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
            
            # Weighted mean of best mu individuals
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * ys[i]
            
            # Update path statistics
            old_x = x.copy()
            x = y_mean
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (L @ ((y_mean - old_x) / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / dim) * chi_n:
                hsig = 1
            else:
                hsig = 0
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 0.001)
            sigma = min(sigma, 10.0)
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (y_mean - old_x) / sigma
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_h * C)
            for i in range(mu):
                diff = (ys[i] - old_x) / sigma
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Search with CMA-ES and Restarts
        # Use 70% of budget for global exploration - leave more room for local refinement
        global_budget = int(budget * 0.70)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        # Run CMA-ES with restarts
        cma_restart_count = 0
        max_cma_restarts = 3
        cma_budget_per_run = global_budget // (max_cma_restarts + 1)
        
        for restart in range(max_cma_restarts + 1):
            if calls >= budget or time.time() > deadline:
                break
            
            # Choose starting point for this CMA-ES run
            if restart == 0:
                x_start = x.copy()
            else:
                if best_x is not None:
                    # Mix of best and random for diversity
                    if rng.random() < 0.3:
                        x_start = best_x + 1.5 * rng.standard_normal(dim)
                    else:
                        x_start = rng.uniform(-5, 5, dim)
                else:
                    x_start = rng.uniform(-5, 5, dim)
                x_start = np.clip(x_start, -5, 5)
            
            # Run CMA-ES with remaining budget for this restart
            remaining_for_cma = min(cma_budget_per_run, budget - calls)
            if remaining_for_cma < 10:
                break
            
            cma_calls_used = 0
            cma_best_x, cma_best_f, cma_calls = self._cma_es(
                safe_func, x_start, remaining_for_cma, dim,
                sigma_init=1.5, pop_size=4,
                max_generations=remaining_for_cma // 5
            )
            cma_calls_used = cma_calls
            
            # Update global best
            if cma_best_f < best_f:
                best_f = cma_best_f
                best_x = cma_best_x.copy()
            
            # Check if we should continue with another restart
            if calls >= budget or time.time() > deadline:
                break

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 15:
                # Nelder-Mead: use a fraction of remaining budget
                # Each iteration uses ~dim+1 calls, so max_iter should be conservative
                max_iter = min(remaining_calls // (dim + 3), 40)
                max_iter = max(8, max_iter)
                
                try:
                    # Initialize simplex around best_x with reasonable spread
                    x0 = best_x.copy()
                    
                    # Create initial simplex with controlled spread
                    spread = 0.3
                    initial_simplex = [x0]
                    for i in range(dim):
                        point = x0.copy()
                        point[i] += spread
                        initial_simplex.append(point)
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-7,
                            'fatol': 1e-7,
                            'adaptive': True,
                            'initial_simplex': initial_simplex
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final Polishing with adaptive (1+1)-ES
        # Sometimes Nelder-Mead gets stuck in a local minimum or plateaus
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                x = best_x.copy()
                fx = best_f  # best_f is already the value at best_x
                
                # Start with moderate sigma and adapt
                sigma = 0.15
                stagnation = 0
                max_stagnation_polish = 15
                improvement_threshold = 1e-12
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    if fy < fx - improvement_threshold:
                        x, fx = y, fy
                        # Successful step: increase sigma slightly
                        sigma = min(sigma * 1.05, 0.5)
                        stagnation = 0
                    else:
                        # Failed step: decrease sigma
                        sigma = max(sigma * 0.80, 0.002)
                        stagnation += 1
                        if stagnation > max_stagnation_polish:
                            # If stuck, try a slightly larger jump or break
                            if sigma > 0.01:
                                sigma = 0.2
                            else:
                                break
                            stagnation = 0
                            max_stagnation_polish = max(8, max_stagnation_polish // 2)

        return best_x, best_f
