import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        if budget < 10:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # CMA-ES Parameters
        # Lambda (population size)
        lam = 12
        # Initial step size
        sigma = 1.5
        # Initial mean
        m = np.zeros(n)
        
        # CMA-ES state
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # Learning rates
        mu = lam // 2  # number of weights
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        # Covariance matrix adaptation
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # History for step size adaptation
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
        
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = m.copy()
        best_f = func(best_x)
        calls_made += 1
        
        while calls_made < budget:
            # Generate lambda offspring
            # Check if we can afford a full generation
            if calls_made + lam > budget:
                # Adjust lambda for the last partial generation if needed, 
                # but simpler to just break if we can't do a full gen? 
                # CMA-ES works better with full generations. 
                # Let's just stop if we can't do a full generation to maintain statistical consistency,
                # or do a partial one. Partial is risky for CMA-ES updates.
                # Given 500 budget and lam=12, 500 % 12 != 0. 
                # 500 - 1 = 499. 499 // 12 = 41 full gens. 41*12 = 492. 
                # We have 7 calls left. 
                # We will do full generations until we can't, then maybe a simple random search or just stop.
                # Stopping is safer for CMA-ES.
                break
            
            # Sample offspring
            Y = rng.standard_normal((lam, n))
            # Whitening: C^0.5 * Y
            # We need to compute sqrt of C. 
            # For efficiency and stability, we can use Cholesky decomposition if C is SPD.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if not SPD
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-12, None)
                L = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            
            X = m + sigma * (L @ Y.T).T
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # Sort offspring
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Compute weighted mean of the best mu offspring
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update step size
            z = (m_new - m) / (sigma * np.linalg.norm(L) if np.linalg.norm(L) > 1e-12 else 1)
            # Note: The standard CMA-ES update for sigma uses the norm of the path p_sigma.
            # p_sigma = (1-cs) * p_sigma + sqrt(cs*(2-cs)*mueff) * C^-0.5 * (m_new - m) / sigma
            # We need C^-0.5.
            # Let's compute C_inv_sqrt via Cholesky of C_inv or eigen.
            # Actually, we can compute C_inv_sqrt from L.
            # If C = L L^T, then C^-1 = L^-T L^-1.
            # C^-0.5 is the matrix square root of C^-1.
            # It's easier to compute C_inv_sqrt once per iteration using eigen or cholesky of C_inv.
            # Let's compute C_inv via Cholesky of C.
            # C_inv = inv(L) @ inv(L).T
            # This is numerically stable.
            try:
                L_inv = np.linalg.inv(L)
                C_inv = L_inv.T @ L_inv
                # C_inv_sqrt = sqrt(C_inv)
                # Use eigen for C_inv_sqrt
                vals_inv, vecs_inv = np.linalg.eigh(C_inv)
                vals_inv = np.clip(vals_inv, 1e-12, None)
                C_inv_sqrt = vecs_inv @ np.diag(np.sqrt(vals_inv)) @ vecs_inv.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ ((m_new - m) / sigma)
            
            # h_sigma condition
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma
            
            # Rank-one and rank-mu update
            C = (1 + c1 * h_sigma * (2 - cc) + cmu - cc - c1) * C \
                + c1 * (p_c @ p_c.T - (1 - cc) * C) \
                + cmu * np.sum(weights[i] * np.outer(X_sorted[i] - m, X_sorted[i] - m) for i in range(mu))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            m = m_new
            
            # Update C_inv for next iteration's whitening
            # We will recompute L in the next loop, so we don't need to store C_inv explicitly unless we want to.
            # But we used C_inv_sqrt. We can just recompute L in the next iteration.

        return best_x, best_f
