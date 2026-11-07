import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES with Hybrid Local Refinement for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    1. Multi-start: Initialize multiple CMA-ES instances from diverse points to handle
       the non-convexity and rotation of the Rosenbrock function.
    2. CMA-ES: Use a robust CMA-ES implementation with eigenvalue clipping for stability.
    3. Hybrid: After the global search phase, run a fast (1+1)-ES from the best found
       point to exploit the local structure of the Rosenbrock valley.
    4. Budget Management: Strict time and call limits to ensure validity and efficiency.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._np_random = np.random.default_rng(42)  # Fixed seed for reproducibility in logic, but instances are distinct

    def _cmaes_run(self, func, x0, sigma0, max_calls, deadline, rng):
        """
        Run a single instance of CMA-ES.
        Returns (best_x, best_f, calls_made).
        """
        n = self.dim
        # CMA-ES parameters tuned for small dimensions
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        m = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls_made = 0
        
        best_x = m.copy()
        best_f = func(m)
        calls_made += 1
        
        # Termination check
        if calls_made >= max_calls or time.time() >= deadline:
            return best_x, best_f, calls_made

        while calls_made < max_calls and time.time() < deadline:
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if calls_made >= max_calls:
                    break
            
            if len(ys) < lambda_pop:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            m_old = m.copy()
            m = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update paths
            yinv = (ys_sorted[:mu] - m_old) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (B @ (D * (m - m_old)))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * (m - m_old)))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            delta_d = (1 + 2 * max(0, mueff / n - 1)) * (np.linalg.norm(pc) ** 2 / n - 1)
            delta = (1 - c_1 - c_mu) * np.eye(n)
            delta += c_1 * (np.outer(pc, pc) + delta_d * np.eye(n))
            for i in range(mu):
                delta += c_mu * weights[i] * np.outer(yinv[i], yinv[i])
                
            # Eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(delta)
            eigvals = np.clip(eigvals, 1e-10, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Termination
            if sigma < 1e-10:
                break
            if np.all(np.abs(ys_sorted[0] - ys_sorted[-1]) < 1e-8):
                break

        return best_x, best_f, calls_made

    def _local_refinement(self, func, x0, max_calls, deadline, rng):
        """
        (1+1)-ES for local refinement.
        """
        n = self.dim
        x = x0.copy()
        fx = func(x)
        calls_made = 1
        sigma = 0.5
        best_x = x.copy()
        best_f = fx
        
        while calls_made < max_calls and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.05
                best_x = x.copy()
                best_f = fx
            else:
                sigma *= 0.95
                if sigma < 1e-6:
                    break
                    
        return best_x, best_f, calls_made

    def __call__(self, func):
        t_start = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = t_start + 320
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        
        # Phase 1: Multi-start CMA-ES
        # Use 3 diverse starts to cover the space
        # Start 1: Origin
        # Start 2: Random point in [-4, 4]
        # Start 3: Random point in [-4, 4]
        
        # We allocate 60% of budget to global search, 40% to local refinement
        global_budget = int(total_budget * 0.6)
        local_budget = total_budget - global_budget
        
        # Split global budget among 3 starts
        per_start_budget = max(10, global_budget // 3)
        
        best_x_global = np.zeros(n)
        best_f_global = float('inf')
        
        rng = np.random.default_rng(123)
        
        starts = [
            np.zeros(n),
            rng.uniform(-4, 4, n),
            rng.uniform(-4, 4, n)
        ]
        
        for i, x0 in enumerate(starts):
            if calls_made >= global_budget or time.time() >= deadline:
                break
                
            # Adjust sigma based on start
            sigma0 = 2.0 if i == 0 else 1.5
            
            x_best, f_best, c_used = self._cmaes_run(
                func, x0, sigma0, per_start_budget, deadline, rng
            )
            calls_made += c_used
            
            if f_best < best_f_global:
                best_f_global = f_best
                best_x_global = x_best.copy()
                
            # If we found a very good solution, we can stop early
            if best_f_global < 1e-4:
                break

        # Phase 2: Local Refinement from the best global point
        if calls_made < total_budget and time.time() < deadline:
            remaining_budget = total_budget - calls_made
            x_best_local, f_best_local, c_used = self._local_refinement(
                func, best_x_global, remaining_budget, deadline, rng
            )
            calls_made += c_used
            
            if f_best_local < best_f_global:
                best_f_global = f_best_local
                best_x_global = x_best_local.copy()

        # Ensure we haven't exceeded the total budget
        # The loops above check budget, but let's be safe.
        # If we somehow went over, we just return the best found.
        # The problem states >500 calls invalidates. We strictly track.
        
        return best_x_global, best_f_global
