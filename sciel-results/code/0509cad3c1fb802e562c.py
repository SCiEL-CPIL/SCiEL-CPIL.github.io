import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan
        n_scan = 15
        points = rng.uniform(-5, 5, (n_scan, self.dim))
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 5.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. Local Refinement using CMA-ES
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        
        # Initialize CMA-ES with the best candidate
        x0 = best_x.copy()
        sigma0 = 2.0
        popsize = 4 + int(3 * np.log(self.dim))
        
        # Manual CMA-ES implementation
        mu = max(1, popsize // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (self.dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + mueff))
        
        C = np.eye(self.dim)
        Cinv = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        x = x0.copy()
        fx = best_f
        best_x_cma = x0.copy()
        best_f_cma = best_f
        
        # Pre-compute matrix square root of C using eigendecomposition
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        
        evals = 0
        while evals < remaining and time.time() < deadline - 2.0:
            # Generate population
            W = np.array([rng.multivariate_normal(np.zeros(self.dim), C, 1).flatten() for _ in range(popsize)])
            W = B @ (D * W).T
            
            Y = x + sigma0 * W
            F = np.array([func(np.clip(y, -5, 5)) for y in Y])
            evals += popsize
            
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f_cma:
                best_f_cma = F[0]
                best_x_cma = Y[0].copy()
            
            # Mean
            m_new = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D * B.T @ (m_new - x)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (evals + 1))) < (1.4 + 2 / (self.dim + 1)) * chiN
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - x) / sigma0
            
            # Covariance matrix update
            if hsig:
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + C - np.eye(self.dim))
            else:
                C = (1 - c_1 - c_mu) * C + c_1 * C
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            for i in range(mu):
                yw = (Y[i] - x) / sigma0
                C = C + c_mu * weights[i] * (np.outer(yw, yw) - delta_hsig * C)
            
            # Re-orthogonalize
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            
            x = m_new
            fx = F[0]
            
            # Step size control
            sigma0 *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Termination criteria
            if np.max(D) / np.min(D) > 1e12:
                break
            if np.linalg.norm(x - best_x_cma) < 1e-8:
                break
            if sigma0 < 1e-12:
                break
        
        if best_f_cma < best_f:
            best_x = best_x_cma
            best_f = best_f_cma

        return best_x, best_f
