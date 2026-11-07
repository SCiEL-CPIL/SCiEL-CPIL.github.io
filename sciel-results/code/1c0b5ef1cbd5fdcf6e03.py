import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Parameters
        lam = 4 + int(3 * np.log(dim))
        if lam % 2 == 1:
            lam += 1
            
        mu = lam // 2
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(lam / 2 + 1) - np.log(i + 1)
        w = w / np.sum(w)
        mu_eff = 2.0 / np.sum(w**2)
        
        c_sigma = (lam / 2.0 + 2.0) / (dim + 4.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((lam / 2.0 - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + lam / dim) / (dim + 4.0 + lam)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + lam))
        
        # State
        m = np.random.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_f = func(m)
        calls_made = 1
        gen = 0
        
        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
        
        while calls_made < budget:
            if time.time() - start_time > 280:
                break
                
            # Generate
            Z = np.random.randn(lam, dim)
            Y = np.zeros((lam, dim))
            for i in range(lam):
                Y[i] = m + sigma * (B @ (D * Z[i]))
                Y[i] = np.clip(Y[i], -5, 5)
                
            F = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget:
                    break
                F[i] = func(Y[i])
                calls_made += 1
                
            if calls_made >= budget:
                # Process last partial or full batch if it finished
                idx = np.argsort(F[:calls_made - 1 - (lam - (calls_made - 1 - gen*lam))] ) # This is getting messy.
                # Simpler: just check all evaluated in this batch
                # We evaluated up to lam points.
                # If we broke out of inner loop, some F[i] might be 0 (uninitialized).
                # We should only consider indices < (calls_made - 1 - gen*lam)
                # Actually, let's just track the number of valid evaluations in this gen.
                pass

            # To keep it simple and robust, let's assume we always complete a generation 
            # unless we hit the exact budget limit.
            # If calls_made >= budget, we break after updating best from this batch.
            
            # Update Best
            # Find min in F
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            if calls_made >= budget:
                break

            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            
            # Weighted Mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * Y_sorted[i]
                
            # Update Mean
            m_new = m + c_c * y_w
            
            # Update Step Size
            y_sigma = B @ (D * y_w)
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) / sigma * y_sigma
            
            # h_sigma calculation
            # The term (1 - (1 - c_sigma)^(2 * gen))
            # gen is the current generation number (starting 0, so this is gen 0, 1, ...)
            # We have just completed generation 'gen'.
            # The formula uses the number of steps taken so far?
            # Standard: h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2 * t)) < (1.4 + 2/(n+1)) * chi_n
            # t is the number of generations.
            # Here t = gen + 1.
            denom = np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (gen + 1)))
            if denom > 0:
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2.0 / (dim + 1.0)) * chi_n) else 0.0
            else:
                h_sigma = 1.0
                
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # Update p_c
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (y_w / sigma)
            
            # Update Covariance
            # h1 = 1 if ||p_c||^2 / (1 - (1 - c_c)^(2 * t)) < (2 + 4/(dim+1)) * dim
            denom_c = np.sqrt(1.0 - (1.0 - c_c) ** (2 * (gen + 1)))
            if denom_c > 0:
                h1 = 1.0 if (np.linalg.norm(p_c)**2 / denom_c**2 < (2.0 + 4.0 / (dim + 1.0)) * dim) else 0.0
            else:
                h1 = 1.0
                
            # C = (1 - c_1 - c_mu + h1 * c_1 * (2 - c_c - c_mu)) * C
            #    + c_1 * (p_c * p_c.T + (1 - h1) * c_c * (2 - c_c) * m_old * m_old.T) 
            #    + c_mu * sum w_i (y_i - m_old) (y_i - m_old).T
            
            # Note: The term (1 - h1) * c_c * (2 - c_c) * m_old * m_old.T is only if h1=0?
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu + h1 * c_1 * (2 - c_c - c_mu)) * C
            #     + c_1 * (p_c p_c.T + (1 - h1) * c_c * (2 - c_c) * m_old m_old.T)
            #     + c_mu * sum_{i=1}^mu w_i (y_(i) - m_old) (y_(i) - m_old).T
            
            # Wait, the term (1 - h1) * c_c * (2 - c_c) * m_old m_old.T is added to the rank-1 update part?
            # Yes, if h1=0, we add a rank-1 term based on the mean to keep C positive definite?
            # Actually, the formula is:
            # C_{new} = (1 - c_1 - c_mu + h1 * c_1 * (2 - c_c - c_mu)) * C_old
            #          + c_1 * (p_c p_c.T + (1 - h1) * c_c * (2 - c_c) * m_old m_old.T)
            #          + c_mu * sum w_i (y_i - m_old) (y_i - m_old).T
            
            # Let's implement this carefully.
            
            # m_old is m
            # y_i are Y_sorted[i]
            
            C_new = (1.0 - c_1 - c_mu + h1 * c_1 * (2.0 - c_c - c_mu)) * C
            C_new += c_1 * (np.outer(p_c, p_c) + (1.0 - h1) * c_c * (2.0 - c_c) * np.outer(m, m))
            
            for i in range(mu):
                diff = Y_sorted[i] - m
                C_new += c_mu * w[i] * np.outer(diff, diff)
                
            # Ensure positive definiteness
            C = C_new
            
            # Eigendecomposition
            D2, B2 = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            D2 = np.maximum(D2, 1e-12)
            D = np.sqrt(D2)
            B = B2
            
            m = m_new
            gen += 1

        return best_x, best_f
