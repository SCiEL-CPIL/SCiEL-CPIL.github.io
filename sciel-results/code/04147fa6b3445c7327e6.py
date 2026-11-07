"""CMA-ES with Rastrigin-aware polish for BBOB f15."""
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Rastrigin-rotated f15.
    
    Strategy:
    1. Use CMA-ES to adapt to the rotation and find the global basin efficiently.
    2. CMA-ES handles the multimodality better than (1+1)-ES due to population-based search
       and covariance adaptation which aligns the search ellipsoid with the function's valleys.
    3. Final polish phase uses coordinate-wise local search to exploit the periodic 
       structure of Rastrigin (period 2) for high-precision convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget_remaining, deadline, rng):
        """
        Run CMA-ES with a fixed budget.
        Returns: (best_x, best_f, calls_made)
        """
        dim = self.dim
        calls_made = 0
        best_f = np.inf
        best_x = x0.copy()
        
        # CMA-ES parameters
        # lambda: population size
        # For dim=5, standard lambda is 4 + floor(3*log(5)) = 4+4=8 or similar.
        # Let's use lambda = 10 for robustness.
        lambda_pop = 10
        mu = lambda_pop // 2  # number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Covariance adaptation
        c1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (1 - cc + 2 * cs / damps)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        m = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        # We need to respect the box [-5, 5]^5. 
        # CMA-ES naturally explores, but we clip. Clipping can distort the distribution.
        # For dim=5 and box [-5,5], starting at 0 with sigma=2.5 is reasonable.
        
        while calls_made < budget_remaining and time.time() < deadline:
            # Generate population
            Y = np.array([rng.multivariate_normal(np.zeros(dim), C) for _ in range(lambda_pop)])
            X = m + sigma * Y
            # Clip to box
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_pop
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Selection
            idx_sorted = np.argsort(F)
            X_parents = X[idx_sorted[:mu]]
            F_parents = F[idx_sorted[:mu]]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, np.newaxis] * X_parents, axis=0)
            
            # Update covariance
            # p_sigma update
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv.dot((m - m_old) / sigma)
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - (1 - cs)**2 * calls_made)) * chi_n) < 1.4 else 0
            
            # p_c update
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * C_sqrt_inv.dot((m - m_old) / sigma)
            
            # C update
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(X_parents - m_old, X_parents - m_old), axis=0) / sigma**2
            
            C = (1 - c1 - cmu) * C + c1 * (rank_one + (1 - h_sigma) * cc * (2 - cc) * np.eye(dim)) + cmu * rank_mu
            
            # Ensure symmetry and positive definiteness (numerical stability)
            C = (C + C.T) / 2.0
            
            # Update C_sqrt via Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
            except np.linalg.LinAlgError:
                # Fallback: reinitialize C to identity if it becomes singular
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Stop criteria
            if sigma < 1e-8:
                break
            if np.all(np.abs(X_parents - m_old) < 1e-8):
                break

        return best_x, best_f, calls_made

    def _rastrigin_polish(self, func, x, budget_remaining, deadline, rng):
        """
        Coordinate-wise local search exploiting Rastrigin's periodicity.
        Rastrigin: 10 + sum(x_i^2 - 10*cos(2*pi*x_i))
        The minima are at integer coordinates (in the rotated frame).
        Since we don't know the rotation, we do a dense local search.
        """
        dim = self.dim
        calls_made = 0
        best_f = func(x)
        best_x = x.copy()
        
        # Try small perturbations in each coordinate
        # Rastrigin has period 2. Local minima are near integers.
        # We try steps that might align with the underlying structure.
        steps = [0.05, 0.1, 0.2, 0.5, 1.0]
        
        for step in steps:
            for i in range(dim):
                for direction in [-1, 1]:
                    y = best_x.copy()
                    y[i] += direction * step
                    y = np.clip(y, -5.0, 5.0)
                    if calls_made >= budget_remaining or time.time() >= deadline:
                        return best_x, best_f, calls_made
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        
        return best_x, best_f, calls_made

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made = 1
        best_f = f0
        best_x = x0.copy()
        
        # Phase 1: CMA-ES
        # Budget allocation: 80% for CMA-ES, 20% for polish
        cma_budget = int(budget * 0.8)
        
        # If we run out of time or budget, skip to polish
        if calls_made < cma_budget and time.time() < deadline:
            # Start CMA-ES from x0
            x_cma, f_cma, cma_calls = self._cma_es(
                func, x0, sigma0=2.5, 
                budget_remaining=cma_budget - calls_made, 
                deadline=deadline, 
                rng=rng
            )
            calls_made += cma_calls
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma.copy()
                
        # Phase 2: Polish
        if calls_made < budget and time.time() < deadline:
            polish_budget = budget - calls_made
            x_polish, f_polish, polish_calls = self._rastrigin_polish(
                func, best_x, 
                budget_remaining=polish_budget, 
                deadline=deadline, 
                rng=rng
            )
            calls_made += polish_calls
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        return best_x, best_f
