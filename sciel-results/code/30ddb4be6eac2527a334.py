"""CMA-ES optimizer for BBOB-style f7 (Step ellipsoidal), dim=5, budget=500."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with initial sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0

        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            return float(func(x))

        # ---- Phase 0: Initial Sampling -----------------------------------
        # Center, axis extremes, and a few random points
        candidates = [np.zeros(d)]
        for i in range(d):
            p1 = np.zeros(d); p1[i] = -4.0; candidates.append(p1)
            p2 = np.zeros(d); p2[i] = 4.0; candidates.append(p2)
        
        rng_init = np.random.default_rng(42)
        for _ in range(5):
            candidates.append(rng_init.uniform(-4, 4, d))
            
        best_x = None
        best_f = np.inf
        for x in candidates:
            if used >= B or time.time() > deadline:
                break
            v = f(x)
            if v < best_f:
                best_f, best_x = v, x.copy()

        # ---- Phase 1: CMA-ES ---------------------------------------------
        # Parameters
        # Population size: usually 4 + 3*log(d)
        lambda_pop = int(4 + 3 * np.log(d))
        # Number of parents
        mu = lambda_pop // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 0.5
        
        # Covariance matrix
        C = np.eye(d)
        
        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Damping parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 - mueff) / (d + 4)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        # Chi-distribution expectation
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
        
        # Main loop
        while used < B and time.time() < deadline:
            # Generate population
            ys = np.zeros((lambda_pop, d))
            xs = np.zeros((lambda_pop, d))
            fs = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample from N(0, C)
                # Use Cholesky decomposition for efficiency and stability
                # L = cholesky(C)
                # y = L @ z
                # But C might not be positive definite due to numerical issues, use eigh
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-12, None)
                L = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
                y = L @ np.random.randn(d)
                ys[i] = y
                xs[i] = best_x + sigma * y
                xs[i] = np.clip(xs[i], LO, HI)
                fs[i] = f(xs[i])
                
                if used >= B or time.time() > deadline:
                    break
            
            if used >= B or time.time() > deadline:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Weighted mean
            m_new = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (m_new - best_x))
            
            # h_sigma condition
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(used+1))) < (1.4 + 2/(d+1)) * chi_d)
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - best_x) / sigma
            
            # Covariance update
            # Rank-one update
            v = (m_new - best_x) / sigma
            C = (1 + c_1 * h_sigma * (1 - h_sigma) - c_1) * C + c_1 * np.outer(v, v)
            
            # Rank-mu update
            # C_mu = sum(weights * outer(y_i - y_bar, y_i - y_bar))
            y_bar = np.sum(weights[:, None] * ys[:mu], axis=0)
            C_mu = np.zeros((d, d))
            for i in range(mu):
                diff = ys[i] - y_bar
                C_mu += weights[i] * np.outer(diff, diff)
            C = (1 - c_1 - c_mu) * C + c_mu * C_mu
            
            # Step size update
            sigma = sigma * np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            # Update mean
            best_x = m_new
            
            # Check for convergence
            if sigma < 1e-12:
                break

        # ---- Phase 2: Coordinate Polish ----------------------------------
        # Simple coordinate descent to clean up
        if used < B and time.time() < deadline:
            for i in range(d):
                if used >= B or time.time() > deadline:
                    break
                lo = max(LO, best_x[i] - 0.5)
                hi = min(HI, best_x[i] + 0.5)
                # Golden section or simple grid
                # Simple grid with 5 points
                for t in np.linspace(lo, hi, 5):
                    if used >= B or time.time() > deadline:
                        break
                    cand = best_x.copy(); cand[i] = t
                    v = f(cand)
                    if v < best_f:
                        best_f, best_x = v, cand.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
