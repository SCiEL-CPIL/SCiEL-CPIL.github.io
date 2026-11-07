"""CMA-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 - mueff / (dim + 1) + 2 * weights[0]) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial state
        x0 = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers a good part of [-5,5]
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Initial evaluation of center
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f
        
        # Initialize population
        # We need to generate lambda points.
        # To save calls, we can start by evaluating the initial population.
        # However, standard CMA-ES evaluates lambda points per generation.
        # Budget is 500. Lam ~ 14. Generations ~ 35.
        
        # Generate initial population
        z = rng.standard_normal((lam, dim))
        # Whitening: B * z * D
        # Initially B=I, D=1, so y = z
        y = z
        x = x0 + sigma * y
        # Clip and evaluate
        x = np.clip(x, -5.0, 5.0)
        
        fvals = np.empty(lam)
        for i in range(lam):
            fi = try_eval(x[i])
            if fi is None:
                # Out of budget, break
                break
            fvals[i] = fi
            
        if call_count >= budget:
             return best_x, best_f

        # Sort
        order = np.argsort(fvals)
        x = x[order]
        fvals = fvals[order]
        
        # Update best
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = x[0].copy()

        # Main Loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # 1. Update mean
            m_old = x0
            m_new = np.sum(weights[:, np.newaxis] * x[:mu], axis=0)
            
            # 2. Update step size
            # ps = (1-cs)ps + sqrt(cs(2-cs)cmu) / sigma * (m_new - m_old)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (m_new - m_old)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            
            # 3. Update covariance matrix
            # pc = (1-cc)pc + hsig*sqrt(cc(2-cc)mueff) * (m_new - m_old)/sigma
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m_old) / sigma
            
            # y = (x - m_old) / sigma
            y = (x[:mu] - m_old) / sigma
            
            # C = (1 - c1 - cmu) C + c1 pc pc^T + cmu sum(wi yi yi^T)
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,jk,il->ikl', y, y, y), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if necessary (for stability)
            C += 1e-12 * np.eye(dim)
            
            # Update B and D (Cholesky)
            try:
                D2, B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(D2))
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                B = np.eye(dim)
                D = np.ones(dim)
                C = np.eye(dim)
            
            # 4. Generate new population
            x0 = m_new
            z = rng.standard_normal((lam, dim))
            # Whitening
            y = np.einsum('ij,jk,kl->il', B, z, np.diag(D))
            x = x0 + sigma * y
            x = np.clip(x, -5.0, 5.0)
            
            # Evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                fi = try_eval(x[i])
                if fi is None:
                    break
                fvals[i] = fi
                
            if call_count >= budget:
                break
                
            # Sort
            order = np.argsort(fvals)
            x = x[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = x[0].copy()

            # Check convergence
            if sigma < 1e-12:
                break

        # Phase 2: Polish with Nelder-Mead if budget remains
        # Use a small budget for polishing
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            x = best_x.copy()
            fx = best_f
            # Simple (1+1) ES with small sigma for final polish
            sigma = 0.01
            for _ in range(min(20, budget - call_count)):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
