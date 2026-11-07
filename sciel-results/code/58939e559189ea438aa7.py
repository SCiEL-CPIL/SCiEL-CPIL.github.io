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
        # Population size lambda. For d=5, lambda=4 or 5 is good.
        # Let's use lambda = 4 for efficiency in small budget.
        lam = 4 
        mu = lam // 2  # number of weights, usually floor(lam/2)
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        # Adaptation parameters
        c_sigma = (lam / 2 + 1) / (d + lam + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + lam / d) / (d + 4 + lam)
        c_1 = 2 / ((d + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + lam))
        
        # Initialization
        # Start at center, as shifts are in [-4,4]
        m = np.zeros(d)
        sigma = 2.0  # Initial step size, box is [-5,5] so width 10. 2 is reasonable.
        C = np.eye(d)
        C_sqrt_inv = np.eye(d)  # Inverse square root of C, initially I
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute constants
        chi_n = np.sqrt(d) * (1 - 1. / (4 * d) + 1. / (21 * d ** 2))
        
        best_x = m.copy()
        best_f = np.inf
        
        # Initial evaluation
        # We must be careful with the budget.
        # We will use a loop that stops when budget is reached.
        
        # To ensure we don't exceed budget, we track evaluations.
        evals_used = 0
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        def evaluate(x):
            nonlocal evals_used
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            evals_used += 1
            return x_clipped, f_val

        # Initial point
        x0, f0 = evaluate(m)
        best_x = x0
        best_f = f0
        
        while evals_used < budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lam, d))
            fvals = np.zeros(lam)
            
            for i in range(lam):
                # Sample from N(0, C)
                z = np.random.randn(d)
                # Transform z to have covariance C
                # We use C_sqrt_inv to update, so we need C_sqrt.
                # Standard CMA-ES implementation often keeps C and uses eigen decomposition or Cholesky.
                # For simplicity and speed in d=5, we can use Cholesky of C.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reinitialize or fix
                    C = np.eye(d)
                    L = np.eye(d)
                    C_sqrt_inv = np.eye(d)
                
                # y = L @ z
                y = L @ z
                x = m + sigma * y
                x, fx = evaluate(x)
                pop[i] = x
                fvals[i] = fx
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update mean
            m_new = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update path states
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_sqrt_inv @ (m_new - m)
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            # We need C_sqrt_inv. 
            # In standard CMA-ES, C_sqrt_inv is updated via the rank-1 and rank-mu updates to C.
            # Actually, it's easier to just compute the inverse square root from C each time if d is small?
            # Or maintain C_sqrt_inv properly.
            # Let's stick to updating C directly and recomputing Cholesky/L for next step.
            # But the ps/pc update requires C_sqrt_inv @ (m_new - m).
            # Let's compute C_sqrt_inv from current C.
            # Since d=5, this is cheap.
            
            # Compute C_sqrt_inv for current C
            # C = L @ L.T
            # C^-1 = L^-T @ L^-1
            # C^(-1/2) = L^-1 @ (L.T @ L)^(-1/2) @ L.T ? No.
            # Simpler: eigendecomposition of C.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Avoid division by zero
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt_inv @ (m_new - m))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            # Update sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - evals_used))) < (1 + 2 / (d + 1)) * chi_n
            if hsig:
                c_sigma_eff = c_sigma
            else:
                c_sigma_eff = 0
            sigma = sigma * np.exp(c_sigma_eff / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update C
            # Rank-mu update
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            # Term for pc
            term_pc = (1 + delta_hsig - c_1) * np.outer(pc, pc)
            
            # Term for top mu
            term_mu = c_mu * np.sum(w[:mu] * np.outer(pop[:mu] - m, pop[:mu] - m) / sigma ** 2, axis=0)
            
            C = (1 + c_1 + c_mu - c_1 - c_mu + 1) * C - term_pc - term_mu
            # Wait, the formula is:
            # C = (1 - c_1 - c_mu + c_1 + c_mu) * C ... no
            # C = (1 - c_1 - c_mu + (1 + 2*c_1) * ... )
            # Standard formula:
            # C = (1 - c_1 - c_mu + c_1 + c_mu) * C is wrong.
            # C = (1 - c_1 - c_mu + 1) * C ...
            # Let's use the explicit formula:
            # C = (1 - c_1 - c_mu + 1) * C is not right.
            # C = (1 - c_1 - c_mu) * C + c_1 * ( (1 + 2*delta_hsig) * outer(pc, pc) + C ) + c_mu * sum(...)
            # Correct formula from Hansen:
            # C = (1 - c_1 - c_mu + c_1 + c_mu) * C ... 
            # Actually:
            # C = (1 - c_1 - c_mu + 1) * C ...
            # Let's use:
            # C = (1 - c_1 - c_mu) * C + c_1 * ( (1 + 2*delta_hsig) * np.outer(pc, pc) + C ) + c_mu * term_mu
            # This simplifies to:
            # C = (1 - c_1 - c_mu + c_1 + c_mu) * C + c_1 * (1 + 2*delta_hsig) * np.outer(pc, pc) + c_mu * term_mu
            # C = C + c_1 * (1 + 2*delta_hsig) * np.outer(pc, pc) + c_mu * term_mu
            # Wait, the "1" in front of C in the rank-1 part comes from the (1 + 2*delta_hsig) * outer(pc,pc) + C term.
            # So: C_new = C_old + c_1 * ( (1 + 2*delta_hsig) * outer(pc, pc) + C_old - C_old ) ... No.
            
            # Let's re-derive carefully.
            # C_{t+1} = (1 - c_1 - c_mu) C_t + c_1 [ (1 + 2 delta_hsig) p_c p_c^T + C_t ] + c_mu sum w_i (y_i y_i^T)
            # C_{t+1} = (1 - c_1 - c_mu + c_1) C_t + c_1 (1 + 2 delta_hsig) p_c p_c^T + c_mu sum ...
            # C_{t+1} = (1 - c_mu) C_t + c_1 (1 + 2 delta_hsig) p_c p_c^T + c_mu sum ...
            
            C = (1 - c_mu) * C + c_1 * (1 + 2 * delta_hsig) * np.outer(pc, pc) + c_mu * term_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness by adding small identity if needed
            # Check eigenvalues
            eigvals_new = np.linalg.eigvalsh(C)
            if np.min(eigvals_new) < 1e-12:
                C += 1e-12 * np.eye(d)
            
            m = m_new
            
            # Check if we should stop early if convergence is reached
            # If sigma is very small, we might be stuck
            if sigma < 1e-8:
                break

        return best_x, best_f
