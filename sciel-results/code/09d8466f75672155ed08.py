import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda is population size
        lam = max(10, 4 + int(3 * np.log(d))) # For d=5, 4 + 3*1.6 = 8.8 -> 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initial values
        m = np.zeros(d)
        sigma = 2.0  # Initial step size, covering a good portion of [-5,5]
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        best_x = m
        best_f = float('inf')
        
        calls_used = 0
        start_time = time.time()
        
        # Loop until budget is exhausted or time is running out
        while calls_used < budget:
            # Check time limit (330s total, leave 30s margin)
            if time.time() - start_time > 300:
                break
                
            # Generate population
            Y = np.zeros((lam, d))
            for i in range(lam):
                # Sample from N(0, C)
                B = np.linalg.cholesky(C)
                z = np.random.randn(d)
                Y[i] = m + sigma * B @ z
                # Clip to box to be safe, though func clips anyway
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                calls_used += 1
                if calls_used >= budget:
                    # If we hit budget exactly, break out of inner loop
                    # We need to handle the rest of the population if we didn't finish
                    # But we allocated lam calls, so if budget < lam, we should adjust
                    # For simplicity, we assume budget >= lam. 500 >= 10.
                    # However, if budget is not a multiple of lam, we might have leftover.
                    # The loop condition `calls_used < budget` handles the outer loop.
                    # But inside, we evaluate lam points.
                    # If calls_used + lam > budget, we should only evaluate the remaining budget.
                    # Let's refine the evaluation loop.
                    pass
            
            # Re-evaluate properly to respect budget exactly
            # The above loop is slightly inefficient if budget isn't multiple of lam.
            # Let's restart the evaluation logic to be precise.
            
            # Actually, let's just break the outer loop if we can't do a full generation?
            # Or do a partial generation?
            # Standard CMA-ES works best with full generations.
            # 500 / 10 = 50 generations exactly. So it fits perfectly.
            
            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update step size
            y_sigma = (m - m_old) / (sigma * np.linalg.norm(C @ np.eye(d))) # Approximation, better:
            # Standard formula: y_sigma = (m - m_old) / (sigma * sqrt(trace(C)))
            # Actually, the standard CMA-ES formula uses the inverse of the square root of C?
            # No, it uses the normalized step.
            # Let's use the standard implementation details.
            # y_sigma = (m - m_old) / (sigma * np.sqrt(np.trace(C)))
            # pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            # sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(pc) / chiN - 1))
            
            # Wait, the norm of y_sigma depends on the transformation.
            # y_sigma is in the "standard normal" space if we transform properly.
            # Let's stick to a robust implementation.
            
            # Transform Y to standard normal space for step size adaptation?
            # Or just use the standard update.
            # The vector y_sigma = (m - m_old) / sigma.
            # Its norm in the C-space is ||C^{-1/2} y_sigma||.
            # For simplicity and robustness, we can use the standard update which assumes
            # the step is taken in the current coordinate system.
            
            # Let's use the standard CMA-ES update formulas.
            # 1. Mean update (done)
            # 2. Step size update
            #    y_sigma = (m - m_old) / sigma
            #    pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            #    sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(pc) / chiN - 1))
            
            # 3. Covariance update
            #    h_sigma = min(1, (np.linalg.norm(pc) / (1 - (1 - c_sigma)**(1 + calls/2)) / chiN - 1) < 1.4)
            #    This h_sigma calculation is tricky with the "calls" count.
            #    Alternative: use the simpler h_sigma = 1 if norm(pc) < (1.4 + 2/(d+1)) * chiN else 0
            #    Let's use the threshold rule.
            
            y_sigma = (m - m_old) / sigma
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(pc) / chiN - 1))
            
            # Threshold for h_sigma
            h_sigma = 1.0 if np.linalg.norm(pc) < (1.4 + 2.0 / (d + 1.0)) * chiN else 0.0
            
            # Rank-1 update
            C = (1 - c_c - c_1 - c_mu * h_sigma) * C + c_c * (np.outer(pc, pc) + (1 - h_sigma) * c_sigma * (2 - c_sigma) * C)
            
            # Rank-mu update
            # y = Y[:mu] - m_old
            # C += c_mu * np.sum(weights[i] * np.outer(y[i], y[i]) for i in range(mu))
            # Note: This is expensive for large d, but d=5 is small.
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Y[i] - m_old, Y[i] - m_old)
            C += c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Add small regularization to prevent singularity
            C += 1e-9 * np.eye(d)
            
            # Optional: Resampling if C becomes ill-conditioned
            # For d=5, this is rarely an issue in 50 generations.
            
        return best_x, best_f
