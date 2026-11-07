"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES with deadline guard for BBOB f17 (Schaffers F7).
    Combines the robust CMA-ES machinery with diverse restarts to handle
    the multiple local minima of the landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline, global_best_f):
        """
        Run a single CMA-ES instance.
        Returns (best_x, best_f, calls_used).
        """
        n = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # CMA-ES parameters
        # For n=5, standard defaults are good.
        # Population size lambda. Let's use lambda = 4 + floor(3 ln n) = 4 + 4 = 8? Or smaller for speed?
        # With budget 500 and restarts, smaller population is better.
        # Let's use lambda = 6, mu = 3.
        lam = 6
        mu = lam // 2  # 3
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 2) / (mueff + 2) - 1) + c_sigma)
        c_c = (4 + mueff / n) / (n + 4 + mueff)
        c_mu = 2 * ((mueff - 2 + 2 / mueff) * c_c - 1) / (n + 2)
        
        # Precompute
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_best = x.copy()
        f_best = global_best_f
        # Note: We don't evaluate x0 here if it was already evaluated or if we just start.
        # But to be safe, let's assume x0 is a candidate. 
        # However, the caller manages the budget. 
        # Let's say this function consumes evaluations.
        # We start with x as the current mean. We don't evaluate x itself in the loop unless it's in the population.
        # Standard CMA-ES generates a population around x.
        
        calls = 0
        start_time = time.time()

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Sample from N(0, C)
            # Use eigen decomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                D = np.eye(n)
                C = np.eye(n)

            Z = rng.standard_normal((lam, n))
            X = x + sigma * (Z @ D.T) # Note: D is symmetric, so D.T = D. But Z @ D.T is correct for (Z_i * D)
            # Actually, if Z is (lam, n), and D is (n, n), then Z @ D.T gives (lam, n) where each row is z_i^T D.
            # We want x + sigma * D z_i. Since D is symmetric, D z_i = z_i^T D.
            
            # Clip to box? The task says func clips. But for CMA-ES, it's better to let it go out and let func clip.
            # However, if we clip X here, the covariance update might be biased.
            # Standard practice: don't clip X, let func clip.
            
            F = np.empty(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(X[i])
                calls += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = X[i].copy()
            
            if calls < lam:
                break

            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update mean
            m_old = x.copy()
            x = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update paths
            y = (x - m_old) / sigma
            # h_sigma check
            # chi_n is E[||N(0,I)||]
            # We need the norm of the evolution path in the transformed space?
            # Standard CMA-ES:
            # y = (m - m_old) / sigma
            # p_sigma = (1-c_sigma) p_sigma + c_sigma^{0.5} (2-c_sigma)^{0.5} mueff^{0.5} y
            # But y is in the current coordinate system.
            # The h_sigma condition uses ||p_sigma|| / chi_n.
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            
            # h_sigma
            # The condition is: ||p_sigma|| / chi_n < (1.4 + 2/(n+1)) * chi_n
            # Wait, the threshold is usually (1.4 + 2/(n+1)) * E[||N(0,I)||]
            # E[||N(0,I)||] is chi_n.
            # So threshold is (1.4 + 2/(n+1)) * chi_n.
            # And we compare ||p_sigma|| / chi_n < threshold? No.
            # The condition is ||p_sigma|| < (1.4 + 2/(n+1)) * chi_n * chi_n?
            # Let's check Hansen's paper.
            # h_sigma = 1 if ||p_sigma|| / chi_n < (1.4 + 2/(n+1)) * chi_n
            # So ||p_sigma|| < (1.4 + 2/(n+1)) * chi_n^2.
            
            h_sigma = (np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * chi_n)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_mu * delta_h - c_mu) * C
            C += c_mu * np.outer(p_c, p_c)
            
            # Rank-mu update
            Y = (X_sorted[:mu] - m_old[None, :]) / sigma
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(Y[i], Y[i])
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C = C + 1e-12 * np.eye(n)

        return x_best, f_best, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s margin
        
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # Strategy:
        # 1. Initial diverse sampling to find good basins.
        # 2. Run CMA-ES from the best points.
        
        # Phase 1: Random Sampling / Exploration
        # Use ~20% of budget for initial exploration?
        # Or just start CMA-ES with diverse starts.
        
        # Let's try 4 restarts.
        # Each restart gets budget/4.
        
        n_restarts = 4
        per_restart_budget = budget // n_restarts
        
        # Diverse starting points
        starts = [
            np.zeros(n),
            rng.uniform(-4, 4, n),
            rng.uniform(-4, 4, n),
            rng.uniform(-4, 4, n),
        ]
        
        # Sort starts by initial function value?
        # Let's evaluate each start first.
        start_fvals = []
        for s in starts:
            if total_calls >= budget:
                break
            fs = func(s)
            start_fvals.append((fs, s))
            total_calls += 1
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
        
        if not start_fvals:
            return best_x, best_f
            
        # Sort by fitness
        start_fvals.sort()
        
        # Run CMA-ES from the best starts
        for i, (fs, s) in enumerate(start_fvals):
            if total_calls >= budget or time.time() > deadline:
                break
            
            # Allocate remaining budget to this restart
            remaining_budget = budget - total_calls
            # If this is the last restart, give it all.
            # Otherwise, give it a chunk.
            if i == len(start_fvals) - 1:
                alloc = remaining_budget
            else:
                alloc = per_restart_budget
                if alloc > remaining_budget:
                    alloc = remaining_budget
            
            if alloc < 10:
                continue
                
            x_best, f_best, calls_used = self._cma_es(
                func, s, 1.5, alloc, rng, deadline, best_f
            )
            
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
        # If we have budget left, refine from the best point
        if total_calls < budget and time.time() < deadline and best_x is not None:
            remaining = budget - total_calls
            if remaining > 20:
                x_best, f_best, calls_used = self._cma_es(
                    func, best_x, 0.5, remaining, rng, deadline, best_f
                )
                total_calls += calls_used
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
                    
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
