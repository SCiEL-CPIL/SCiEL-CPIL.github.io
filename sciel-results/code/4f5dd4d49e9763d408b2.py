import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # State
        x_mean = np.zeros(n) # Hint: near 0
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = 0
        best_x = x_mean
        best_f = float('inf')
        
        # Initial evaluation
        f0 = func(x_mean)
        calls_made += 1
        best_f = f0
        best_x = x_mean.copy()
        
        # Check time/budget
        if calls_made >= self.budget:
            return best_x, best_f
            
        while calls_made < self.budget:
            if time.time() >= deadline:
                break
                
            # Generate population
            zs = np.random.normal(size=(lam, n))
            # Transform: y = B * (D * z)
            # z is (lam, n). D is (n,). B is (n, n).
            # We want y_i = sum_j B_ij D_j z_ij ? No.
            # C = B D^2 B^T.
            # Sample y = B D z.
            # z ~ N(0, I).
            # y = (z * D[None, :]) @ B.T
            ys = (zs * D[None, :]) @ B.T
            pop = x_mean + sigma * ys
            
            # Clip
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fs = np.empty(lam)
            for i in range(lam):
                fs[i] = func(pop[i])
                calls_made += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = pop[i].copy()
                
                if calls_made >= self.budget:
                    break
                    # If we break here, we might have partial population.
                    # CMA-ES needs full population for update.
                    # If we can't finish population, break out of main loop.
                    # Let's check budget before starting population.
            
            if calls_made >= self.budget:
                break

            # Sort
            order = np.argsort(fs)
            pop_sorted = pop[order]
            fs_sorted = fs[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update step size
            # y = (x_mean - x_mean_old) / sigma
            y = (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            hsig = (np.sqrt(1 - (1 - c_sigma)**2) * np.sqrt(n + 1) / np.sqrt(n) > np.sqrt(n + 1) / np.sqrt(n))
            # Standard hsig check:
            # hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made//lam + 1))) < (1 + 2/(n+1)) * chi_n)
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            # Approximation for hsig
            hsig = (np.linalg.norm(ps) < (1 + 2/(n+1)) * chi_n)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Update covariance
            # pc = (1 - c_c) * pc + hsig * sqrt(c_c * (2 - c_c) * mueff) * y
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Rank-one update
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            # Rank-mu update
            # delta_mu = c_mu * (2 - c_mu)
            
            # C = (1 - c_1 - c_mu + delta_hsig) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * I)
            # Wait, the rank-one update term is c_1 * (pc @ pc.T + (1-hsig)*c_c*(2-c_c)*I)
            # And the rank-mu term is c_mu * sum w_i (y_i @ y_i.T)
            
            # Let's use the standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.eye(n))
            #    + c_mu * np.sum(w[:, None, None] * np.einsum('i,ij,ik->ijk', w, y_sorted, y_sorted), axis=0)
            
            # Actually, y_sorted are the normalized steps of the selected parents.
            # y_i = (x_i - x_mean_old) / sigma
            y_parents = (pop_sorted[:mu] - x_mean_old) / sigma
            
            # Rank-mu term
            # sum_i w_i (y_i y_i^T)
            rank_mu_term = np.einsum('i,ij,ik->jk', w, y_parents, y_parents)
            
            # Rank-one term
            rank_one_term = pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.eye(n)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_term + c_mu * rank_mu_term
            
            # Cholesky decomposition for next iteration
            # Ensure positive definiteness
            # Add small regularization if needed
            C += 1e-10 * np.eye(n)
            try:
                L = np.linalg.cholesky(C)
                B, D = np.linalg.qr(L)
                D = np.diag(D)
                # Ensure D is positive
                D = np.abs(D)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity or use previous
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                sigma *= 0.5 # Shrink step size
        
        return best_x, best_f
