import numpy as np
import time

class Optimizer:
    """
    Population-based CMA-ES for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Full CMA-ES implementation to handle rotation and non-separability.
    - Large initial sigma to cover the search space quickly.
    - Strict budget and time management.
    - Fallback to (1+1)-ES if CMA-ES stalls or budget is tight.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline):
        """
        Run CMA-ES from x0 with initial sigma sigma0.
        Returns (best_x, best_f, calls_made).
        """
        n = self.dim
        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(n)))  # Population size
        mu = lambda_pop // 2  # Number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # State
        m = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls_made = 0
        
        # Initial evaluation
        f0 = func(m)
        calls_made += 1
        best_x = m.copy()
        best_f = f0
        
        while calls_made < max_calls and time.time() < deadline:
            # Sample population
            ys = []
            fs = []
            for i in range(lambda_pop):
                z = np.random.standard_normal(n)
                y = m + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                y_clipped = np.clip(y, -5, 5)
                fy = func(y_clipped)
                calls_made += 1
                ys.append(y_clipped)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y_clipped.copy()
                    
                if calls_made >= max_calls:
                    break
            
            if len(ys) < lambda_pop:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            m_old = m.copy()
            m = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update covariance matrix
            yinv = np.zeros((mu, n))
            for i in range(mu):
                yinv[i] = (ys_sorted[i] - m_old) / sigma
            
            # Cumulation path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (B @ (D * (m - m_old)))
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * (m - m_old)))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix (rank-mu + rank-1)
            delta_d = (1 + 2 * max(0, mueff / n - 1)) * (np.linalg.norm(pc) ** 2 / n - 1)
            # Simplified update for speed and stability in this context
            # Full rank-mu update:
            delta = (1 - c_1 - c_mu) * np.eye(n)
            delta += c_1 * (np.outer(pc, pc) + delta_d * np.eye(n))
            delta += c_mu * sum(weights[i] * np.outer(yinv[i], yinv[i]) for i in range(mu))
            
            # Eigenvalue decomposition to maintain positive definiteness
            eigvals, eigvecs = np.linalg.eigh(delta)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-10, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Termination criteria
            if sigma < 1e-10:
                break
            if np.all(np.abs(ys_sorted[0] - ys_sorted[-1]) < 1e-8):
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # Safety margin
        
        # Initial best
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        calls_made = 0
        
        # Phase 1: CMA-ES with large sigma to find basin
        # Rosenbrock in 5D with rotation needs a wide search
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        calls_made += 1
        best_f = f0
        best_x = x0.copy()
        
        # Try a few random starts if origin is bad? 
        # For f9, origin is usually decent but rotation matters.
        # Let's use CMA-ES from origin with sigma=2.0 (covers most of [-5,5] in 2-3 steps)
        
        # If we have budget, run CMA-ES
        if calls_made < self.budget and time.time() < deadline:
            # Allocate 80% of budget to CMA-ES
            cma_budget = int(self.budget * 0.8)
            best_x, best_f, cma_calls = self._cmaes(func, x0, 2.0, cma_budget, deadline)
            calls_made += cma_calls
            
        # Phase 2: Refinement with (1+1)-ES or smaller CMA-ES
        if calls_made < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = self.budget - calls_made
            
            # Use (1+1)-ES for fast local convergence
            while calls_made < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * np.random.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    if sigma < 1e-6:
                        break

        return best_x, best_f
