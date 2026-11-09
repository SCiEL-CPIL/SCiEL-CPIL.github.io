"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for the BBOB f17 (Schaffers F7) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        total_budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es_run(rng, x0, sigma0, budget_left):
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            m = 10  # Population size
            mu = m // 2  # Selection size
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_cholesky = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))
            
            # Initial evaluation
            fx = func(x)
            track_call(x, fx)
            
            pop_x = np.zeros((m, n))
            pop_f = np.zeros(m)
            
            # Initial population
            for i in range(m):
                if time.time() > deadline or calls_used >= total_budget:
                    return
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * C_cholesky @ z, -5, 5)
                pop_x[i] = y
                pop_f[i] = func(y)
                track_call(y, pop_f[i])
                
            order = np.argsort(pop_f)
            
            # Main loop
            while calls_used + m <= total_budget and time.time() < deadline:
                # Update mean
                x_new = np.sum(weights[:, None] * pop_x[order[:mu]], axis=0)
                
                # Update evolution paths
                # ps update
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(C_cholesky) * (C @ (C_cholesky.T @ (x_new - x))) / sigma
                # Note: The standard update is ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mueff) * B^-1 * (x_new - x) / sigma
                # where B is the Cholesky factor of C.
                # Let's use the more standard formulation:
                # h_sigma = (||ps|| / (1 - (1-c_sigma)^{1/gen})) / (chi_n * sqrt(1 - (1-c_sigma)^{2/gen}))
                # For simplicity and robustness in short runs, we use the basic update.
                
                # Actually, let's stick to the standard robust update:
                # y = (x_new - x) / sigma
                y = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * np.linalg.solve(C_cholesky.T, y)
                
                h_sigma = (np.linalg.norm(ps) / chi_n) < (1.4 + 2 / (n + 1)) * (1 - (1 - c_sigma) ** 100)
                
                # pc update
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - delta_h * C)
                
                # Rank-mu update
                y_diff = (pop_x[order[:mu]] - x) / sigma
                # Transform to whitened space? No, just use the raw differences weighted.
                # The standard rank-mu update is:
                # C += c_mu * sum( w_i * y_i * y_i^T )
                # But y_i are in the original space. We need to project them or use the formula:
                # C += c_mu * sum( w_i * (B @ y_i) * (B @ y_i)^T ) is not quite right.
                # The correct formula uses the transformed points.
                # Let's use the simpler approximation often found in implementations:
                # C += c_mu * sum( w_i * np.outer(y_i, y_i) )
                # This is only correct if C is identity. For general C, we should use:
                # C += c_mu * sum( w_i * np.outer(B @ y_i, B @ y_i) )? No.
                
                # Let's use the standard update:
                # C += c_mu * sum_{i=1}^mu w_i * y_i * y_i^T
                # where y_i = (x_i - x_old) / sigma.
                # This is actually an approximation. The exact update involves the Cholesky factor.
                # Given the small dimension (5), we can afford the exact update.
                # y_i = (x_i - x) / sigma
                # The update is C <- (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - h_sigma) * C) + c_mu * sum(w_i * y_i * y_i^T)
                # Wait, the term is sum(w_i * (B y_i) (B y_i)^T) ? No.
                # The standard CMA-ES update for C is:
                # C <- (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - h_sigma) * C) + c_mu * sum_{i=1}^mu w_i * y_i * y_i^T
                # This is valid if y_i are the step vectors in the search space.
                
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(y_diff[i], y_diff[i])
                    
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update Cholesky factor
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, add small regularization
                    C += 1e-6 * np.eye(n)
                    C_cholesky = np.linalg.cholesky(C)
                    
                x = x_new
                fx = pop_f[order[0]]
                track_call(x, fx)
                
                # Generate new population
                for i in range(m):
                    if time.time() > deadline or calls_used >= total_budget:
                        return
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * C_cholesky @ z, -5, 5)
                    pop_x[i] = y
                    pop_f[i] = func(y)
                    track_call(y, pop_f[i])
                    
                order = np.argsort(pop_f)

        # Strategy:
        # 1. Initial exploration to find a good starting point
        # 2. Main CMA-ES run
        # 3. Restart CMA-ES if budget remains
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Exploration
        explore_budget = min(20, total_budget // 10)
        for _ in range(explore_budget):
            if time.time() > deadline or calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)
            
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            
        remaining = total_budget - calls_used
        
        # Phase 2: Main CMA-ES
        if remaining > 20:
            cma_es_run(rng, best_x, 1.0, remaining)
            
        remaining = total_budget - calls_used
        
        # Phase 3: Restart if time/budget allows
        if remaining > 20 and time.time() < deadline - 10:
            # Pick a random point or a point far from best
            x2 = rng.uniform(-5, 5, dim)
            cma_es_run(rng, x2, 2.0, remaining)
            
        remaining = total_budget - calls_used
        
        # Phase 4: Another restart if possible
        if remaining > 20 and time.time() < deadline - 5:
            x3 = rng.uniform(-5, 5, dim)
            cma_es_run(rng, x3, 1.5, remaining)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
