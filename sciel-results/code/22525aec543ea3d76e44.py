import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        def _check_time():
            return time.time() - start_time < timeout
        
        # CMA-ES Parameters - tuned for d=5
        lam = 8  # population size
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))
        
        # Initialize
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        # PC and PS for path evolution
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # We need to track the best x and f found globally.
        best_x = None
        best_f = np.inf
        
        n_evals = 0
        it = 0
        
        # Phase 1: CMA-ES exploration
        while n_evals + lam <= int(0.6 * budget) and _check_time():
            # Sample
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            n_evals += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            # Sort
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            
            # Update mean
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])
            
            # Path evolution sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1
            
            # Path evolution covariance
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))
            
            # Update covariance
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            # Step size adaptation
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Stop if sigma is very small
            if sigma < 1e-5:
                break

        # Phase 2: CMA-ES polish with smaller population and tighter sigma
        if best_x is not None and n_evals < budget and _check_time():
            # Reset for polish phase
            x_mean = best_x.copy()
            sigma = max(sigma * 0.1, 1e-4)
            C = np.eye(d)
            B = np.eye(d)
            D = np.ones(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            it = 0
            
            lam2 = 6
            mu2 = lam2 // 2
            w2 = np.log(mu2 + 0.5) - np.log(np.arange(1, mu2 + 1))
            w2 = w2 / np.sum(w2)
            mueff2 = 1.0 / np.sum(w2**2)
            
            cc2 = (4 + mueff2 / d) / (d + 4 + 2 * mueff2 / d)
            cs2 = (mueff2 + 2) / (d + mueff2 + 5)
            c12 = 2 / ((d + 1.3)**2 + mueff2)
            cmu2 = min(1 - c12, 2 * ((mueff2 - 2 + 1 / mueff2) / ((d + 2)**2 + mueff2)))
            
            while n_evals + lam2 <= budget and _check_time():
                Z = np.random.randn(lam2, d)
                Y = x_mean + sigma * (Z * D) @ B.T
                Y = np.clip(Y, -5, 5)
                
                F = np.array([func(y) for y in Y])
                n_evals += lam2
                
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = Y[min_idx].copy()
                
                sorted_idx = np.argsort(F)
                Y_sorted = Y[sorted_idx]
                
                y_old = x_mean.copy()
                x_mean = np.dot(w2, Y_sorted[:mu2])
                
                ps = (1 - cs2) * ps + np.sqrt(cs2 * (2 - cs2) * mueff2) / sigma * (B.T @ (D * (x_mean - y_old)))
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs2)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc2
                it += 1
                
                if hsig:
                    pc = (1 - cc2) * pc + cc2 / sigma * (B.T @ (D * (x_mean - y_old)))
                
                delta_hebbian = c12 * (np.outer(ps, ps) + (1 - hsig) * (cc2 * (2 - cc2) * np.outer(pc, pc)))
                
                for i in range(mu2):
                    delta_hebbian += (cmu2 * w2[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                    
                C = (1 - c12 - cmu2) * C + delta_hebbian
                C = (C + C.T) / 2
                
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D2, 1e-20, None))
                
                chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
                sigma *= np.exp(cs2 / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-8, 1.0)
                
                if sigma < 1e-7:
                    break

        # Phase 3: Coordinate-wise refinement (cheap and effective for ellipsoidal)
        if best_x is not None and n_evals < budget and _check_time():
            x = best_x.copy()
            f_cur = best_f
            # Try small steps in each coordinate
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for ss in step_sizes:
                if n_evals >= budget or not _check_time():
                    break
                for i in range(d):
                    for direction in [-1, 1]:
                        if n_evals >= budget or not _check_time():
                            break
                        x_try = x.copy()
                        x_try[i] += direction * ss
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        n_evals += 1
                        if f_try < f_cur:
                            f_cur = f_try
                            x = x_try.copy()
            if f_cur < best_f:
                best_f = f_cur
                best_x = x.copy()

        # Final check
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_evals += 1
            
        return best_x, best_f
