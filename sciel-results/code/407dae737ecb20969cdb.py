import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 4 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute C^{-1/2} and C^{1/2} initially (Identity)
        # We will update C and its square root each generation
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls_made = 1
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls_made >= budget: break
                if time.time() > deadline: break
                
                # Sample from N(0, C)
                # Use Cholesky or eigen decomposition of C
                # For dim=5, this is cheap.
                # Ensure C is symmetric positive definite
                # Symmetrize C
                C = (C + C.T) / 2.0
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if not SPD
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                
                z = rng.standard_normal(dim)
                y = L @ z
                x_trial = m + sigma * y
                x_trial_clipped = np.clip(x_trial, -5, 5)
                
                f_val = func(x_trial_clipped)
                calls_made += 1
                
                ys.append(y)
                xs.append(x_trial_clipped) # Use clipped for evaluation, but ideally update m with unclipped?
                # Task says func clips. The "true" function is defined on the box.
                # Usually, for constrained CMA-ES, one handles boundaries carefully.
                # Given the simple constraint and interior optimum, clipping is likely fine.
                # We will update m using the clipped points to stay within bounds.
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_trial_clipped.copy()
            
            if not ys:
                break
            
            # Sort by fitness
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            m_new = m + sigma * np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * (C^{-1/2} (m_new - m))
            # We need C^{-1/2}. Let's approximate or compute.
            # For dim=5, computing inverse sqrt is ok.
            # Actually, standard CMA-ES uses:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (m_new - m) / sigma
            
            # Compute C^{-1/2}
            # Since we have L = C^{1/2}, C^{-1/2} = L^{-1}
            # Solving L v = (m_new - m) / sigma
            try:
                C_inv_sqrt_m = np.linalg.solve(L, (m_new - m) / sigma)
            except:
                C_inv_sqrt_m = (m_new - m) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_m
            
            # Update sigma
            # h_sigma = (||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 gen})) < c_1 + 1.4
            # This requires generation count. Let's simplify or implement properly.
            # For short runs, a simpler update might suffice, but let's try to be accurate.
            # Approximation:
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            # h_sigma condition
            # We need generation counter for the denominator.
            # Let's assume gen is small.
            # h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * chi_N else 0
            # This is a simplification.
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} (m_new - m) / sigma
            # Note: The vector in p_c update is the same as in p_sigma update (the step in normalized space)
            h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * chi_N else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt_m
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            # y_i are the normalized steps (before sigma multiplication)
            
            y_rank1 = np.outer(p_c, p_c)
            if h_sigma == 0:
                y_rank1 += c_c * (2 - c_c) * C
            
            y_rankmu = np.zeros((dim, dim))
            for i in range(mu):
                y_rankmu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * y_rank1 + c_mu * y_rankmu
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / (2 * (dim + 1)) * (||p_sigma||^2 / dim - 1))
            # Wait, the standard formula is:
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_N - 1))
            # d_sigma = (dim + 2) + c_sigma - 1 ? No.
            # d_sigma = (1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1)) / (dim + 4 + mu_eff) ? No, that's c_sigma.
            # d_sigma is usually dim + 2 + c_sigma - 1.
            # Let's use the standard:
            d_sigma = dim + 2 + c_sigma - 1
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            m = m_new
            # Clip m to box to keep it in valid range for next iterations?
            # The problem says func clips. But keeping m inside is good practice.
            m = np.clip(m, -5, 5)

        return best_x, best_f
