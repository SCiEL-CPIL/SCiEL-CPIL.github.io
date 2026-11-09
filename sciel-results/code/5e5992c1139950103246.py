import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(1, mu + 1):
            w[i-1] = np.log(lambda_pop/2 + 0.5) - np.log(i)
        w /= w.sum()
        mueff = 1.0 / np.dot(w, w)
        
        # Adaptation rates
        cs = (1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1)) / (d + 2 + mueff / 2)
        cc = (4 - lambda_pop / mueff + 2) / (d + 4 + 2 / mueff)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        # Initial state
        mean = np.zeros(d)
        sigma = 2.5
        C = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        best_x = mean.copy()
        best_f = np.inf
        count = 0
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the instance
        
        while count < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Use Cholesky decomposition for C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or regularize
                C = np.eye(d)
                L = np.eye(d)
            
            ys = rng.standard_normal((lambda_pop, d))
            zs = np.dot(ys, L.T)
            xs = mean + sigma * zs
            
            # Clip to box [-5, 5] to respect constraints and avoid distortion
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            count += len(fs)
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
                
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            
            # Update mean
            y_w = np.dot(w, xs_sorted[:mu].T)
            m_old = mean
            mean = m_old + sigma * y_w
            
            # Update step size control variable ps
            # y_sigma = (1/sigma) * C^{-1/2} * (x_w - m_old)
            # Note: x_w is the weighted sum of the sorted x's? 
            # Standard CMA-ES:
            # x_w = sum w_i x_{i:lambda}
            # y_sigma = (1/sigma) * C^{-1/2} * (x_w - m_old)
            
            x_w = np.dot(w, xs_sorted[:mu].T)
            # C^{-1/2} is inv(L) if C = L L^T
            try:
                inv_L = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                inv_L = np.eye(d)
            
            y_sigma = np.dot(inv_L, (x_w - m_old) / sigma)
            
            ps = (1 - cs) * ps + cs * y_sigma
            
            # Heuristic hsig
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count + 1) - 1))) < \
                   (1 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            
            if hsig:
                pc = (1 - cc) * pc + cc * y_sigma
            
            # Update covariance
            v = np.dot(ps, ps) / d - 1
            delta_hs = (1 - hsig) * cc * (2 - cc)
            
            # C = (1 + c1(2 - c1 + cmu) - c1 v) C + c1 ps ps^T + cmu sum w_i (x_i - m_old)(x_i - m_old)^T / sigma^2
            C = (1 + c1 * (2 - c1 + cmu) - c1 * v) * C
            C += c1 * np.outer(ps, ps)
            
            diff = (xs_sorted[:mu] - m_old).T
            C += cmu * (w @ (diff @ diff.T)) / (sigma ** 2)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            # d_sigma is typically 1.0 or d/2 + 1
            d_sigma = 1.0 
            sigma = sigma * np.exp(cs / d_sigma * (np.linalg.norm(ps) / np.sqrt(d) - 1))
            
            # Clamp sigma
            if sigma < 1e-10:
                break
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
