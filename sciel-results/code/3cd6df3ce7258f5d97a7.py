import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # For 5D, lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8 is a reasonable choice
        # Or standard: lambda = 2 * dim = 10. Let's use 8 for efficiency.
        lambda_ = 8
        mu = lambda_ // 2  # Number of weights
        
        # Weights: log-linear
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size
        sigma = 2.0 # Start with a reasonable step size for [-5,5] box
        
        # Mean
        m = np.zeros(dim)
        
        # Covariance Matrix
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        # We need to update B and D such that C = B B^T and D = 1/sqrt(diag(B^T B))
        # Initial B = I, D = 1
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Update rates
        cs = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cmu = min(1 - cs, 2 * (mueff - 2) / ((dim + 2)**2 - 2))
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        
        # Initial evaluation of mean
        f_m = evaluate(m)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample lambda offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                # Sample y ~ N(0, C)
                # y = B @ (D * z)
                z = np.random.normal(size=dim)
                y = B @ (D * z)
                
                # x = m + sigma * y
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                
                if calls_made >= budget:
                    break
                    
                f_val = evaluate(x)
                ys.append(y)
                xs.append(x)
                fs.append(f_val)
                
            if len(xs) < lambda_:
                break
                
            # 2. Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # 3. Update mean
            y_weighted = np.sum([w[i] * ys[i] for i in range(mu)], axis=0)
            m_new = m + sigma * y_weighted
            m_new = np.clip(m_new, -5, 5)
            
            # 4. Update evolution paths
            # ps = (1-cs)ps + sqrt(cs(2-cs)cmu) * (1/sigma) * y_weighted
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_weighted / sigma
            
            # hsig = ||ps|| / sqrt(1-(1-cs)^{2/(lambda+1)}) / sqrt(1-cs^2)
            # Approximation for hsig check
            norm_ps = np.linalg.norm(ps)
            # chi_n is expected norm of N(0, I)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            
            # If ps is not too large, update pc
            # hsig is usually 1 if we haven't diverged, 0 if we have
            # Simplified hsig: 1 if norm_ps < 1.4 * chi_n * (1 + cs) else 0
            # More accurate:
            # denom = np.sqrt(1 - (1-cs)**(2.0/(lambda_ + 1))) / np.sqrt(1 - cs**2)
            # hsig = 1 if norm_ps / denom < 1.4 * chi_n * (1 + cs) else 0
            
            # For simplicity and stability in this short budget, let's use a standard hsig approximation
            # or just always update pc if ps is small enough relative to chi_n
            # Let's use the standard formula
            # Note: In many implementations, hsig is 1 or 0.
            
            # Calculate hsig
            # We need to be careful with the denominator
            # Let's use a simplified check: if norm_ps is not exploding
            if norm_ps < 1.5 * chi_n:
                hsig = 1.0
            else:
                hsig = 0.0
                
            # pc = (1-cs)pc + hsig * sqrt(cs(2-cs)cmu) * (1/sigma) * (m_new - m)
            # Wait, the formula is: pc = (1-cs)pc + hsig * sqrt(cs(2-cs)cmu) * (1/sigma) * (m_new - m)
            # Actually, the standard formula uses y_weighted for ps, and (m_new-m)/sigma for pc?
            # No, pc uses the difference of means.
            # pc = (1-cs)pc + hsig * sqrt(cs(2-cs)cmu) * (m_new - m) / sigma
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (m_new - m) / sigma
            
            # 5. Update Covariance Matrix C
            # C = (1 - cmu - cmu_c) C + cmu * sum(w_i y_i y_i^T) + cmu_c * pc pc^T
            # cmu_c = cmu * cs / (2 - cs)
            
            # Calculate weighted sum of outer products
            # y_i are the weighted sum components? No, y_i are the sorted y's.
            # The term is sum_{i=1..mu} w_i * y_i * y_i^T
            # Note: y_i here are the y vectors of the top mu offspring.
            
            y_outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                y_outer_sum += w[i] * np.outer(ys[i], ys[i])
                
            cmu_c = cmu * cs / (2 - cs)
            
            C_new = (1 - cmu - cmu_c) * C + cmu * y_outer_sum + cmu_c * np.outer(pc, pc)
            
            # Ensure symmetry and positive definite
            C_new = (C_new + C_new.T) / 2.0
            
            # Update B and D (Eigen decomposition)
            # C = B B^T
            # We need B and D such that C = B (D D^T) B^T? No, C = B B^T.
            # Standard CMA-ES uses C = B (D^2) B^T? No.
            # C = B B^T. And we sample y = B D z.
            # So C = B D^2 B^T.
            # Eigen decomposition: C = V diag(lambda) V^T.
            # So B = V, D = sqrt(diag(lambda)).
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C_new)
                # Clip negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                # Fallback if eigen decomposition fails
                break
                
            # 6. Update sigma
            # sigma = sigma * exp(c_sigma/d_sigma * (||ps||/chi_n - 1))
            sigma = sigma * np.exp((c_sigma / ds) * (norm_ps / chi_n - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            m = m_new
            
            # Check convergence
            if np.linalg.norm(sigma * D) < 1e-10:
                break

        return best_x, best_f
