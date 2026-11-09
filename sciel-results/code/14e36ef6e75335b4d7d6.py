import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Search
        # We have 500 calls. Let's use 125 for the grid.
        # 5^5 = 3125 is too many.
        # Let's try a random sampling approach which is more efficient for high-dim.
        # But the directive suggests a coarse grid scan.
        # In 5D, a grid is hard. 5 points per dim -> 3125.
        # 4 points per dim -> 1024.
        # 3 points per dim -> 243. This is feasible.
        # Let's do 3 points per dimension: -5, 0, 5.
        # Actually, let's use Latin Hypercube Sampling for better coverage with fewer points.
        # Or just random sampling.
        # The directive says "coarse grid scan ... using ~400 calls".
        # Let's do 256 random points first to identify a good region.
        
        # Actually, let's stick to the directive: "coarse grid scan".
        # In 5D, a structured grid is inefficient. Let's interpret "coarse grid" as
        # a stratified random sample or a low-resolution grid.
        # Let's use 125 random samples (5^3? no).
        # Let's use 250 random samples.
        
        # Re-evaluating: CMA-ES is very strong. The current score is 1.93.
        # The inspiration 1 (2.12) used CMA-ES.
        # The directive suggests: 400 calls for global, 100 for local.
        # Let's try that.
        
        # Global Phase: 400 calls
        # Use a set of random points, maybe with some structure.
        # Or use a simple (1+1)-ES with large step size?
        # Let's use random sampling + best point tracking.
        
        # Actually, a better global strategy for 5D is to sample 100 random points,
        # find the best, then run CMA-ES from there with a small budget?
        # But CMA-ES needs ~10-20 * dim calls to converge. 50 calls is tight.
        
        # Let's try:
        # 1. Sample 100 random points. (100 calls)
        # 2. Take the best point.
        # 3. Run CMA-ES with remaining 400 calls.
        
        # This is a standard hybrid approach.
        
        # Phase 1: Random Initialization
        n_init = 100
        for _ in range(n_init):
            if calls >= budget or time.time() >= deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            eval_point(x)
        
        # Phase 2: CMA-ES
        # Start CMA-ES from best_x found in Phase 1.
        m = best_x.copy()
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        sigma = 1.0  # Start with a moderate step size, CMA-ES will adapt
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        while calls < budget:
            if time.time() >= deadline:
                break
            
            # Sample population
            Z = np.random.standard_normal((lam, dim))
            
            # Compute C^1/2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = Z @ C_sqrt.T
            X = m + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([eval_point(x) for x in X])
            
            # Sort
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            F_sorted = F[idx_sort]
            
            # Update Mean
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update Step Size
            c_1_mu = c_1 + c_mu
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + c_c * delta
            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
            
            C_new = (C_new + C_new.T) / 2
            m = m_new
            C = C_new
            
            # Early stopping if sigma is very small or no improvement for a while?
            # CMA-ES handles this, but we have a hard budget.
            
        return best_x, best_f
