"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer: adapts covariance matrix to align with function valleys."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                fx = float(func(x))
            except:
                fx = np.inf
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Population size lambda = 4 + floor(3 log n)
        lam = 4 + int(3 * np.log(dim))
        # For dim=5, lam = 4 + 3*1.6 = 8. But larger is often better for exploration.
        # Let's use lam=16 as per inspiration for better covariance estimation.
        lam = 16
        
        # Step size sigma
        # Initial sigma: 1.0 is standard.
        sigma = 1.0
        
        # Mean (start at 0, or maybe best from a quick random check? 
        # CMA-ES is usually robust from 0, but a quick check helps if shifted far.
        # However, CMA-ES with sigma=1 covers [-5,5] reasonably in ~3-5 steps.
        # Let's start at 0.
        mean = np.zeros(dim)
        
        # Covariance matrix C (initially Identity)
        C = np.eye(dim)
        
        # Inverse Cholesky factor of C (for sampling)
        C_inv_chol = np.eye(dim)
        
        # Learning rates
        # mu = floor(lam/2)
        mu = lam // 2
        # Weights for the mu best individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        # mu_eff
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rate for step size (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # Learning rate for covariance matrix
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
        
        # Damping for Hessian update
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 2) / (dim + 2)) - 1) + c_mu
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cache of evaluated points to avoid redundant calls?
        # CMA-ES generates unique points usually, but due to float precision or symmetry,
        # collisions are rare. We'll skip explicit caching for simplicity and speed,
        # relying on the fact that CMA-ES points are distinct.
        
        # Pre-compute weights
        # We need to sort the population by fitness and take the top mu.
        
        while total_calls < budget and time.time() < deadline:
            # 1. Sample lambda points
            points = []
            for i in range(lam):
                # Check budget
                if total_calls >= budget or time.time() > deadline:
                    break
                
                # Standard normal sample
                z = np.random.randn(dim)
                
                # Transform to x
                # x = mean + sigma * C_inv_chol.T @ z
                x = mean + sigma * (C_inv_chol.T @ z)
                
                # Evaluate
                fx = safe_call(x)
                points.append((fx, x))
            
            if len(points) < lam:
                break
            
            # 2. Sort by fitness
            points.sort(key=lambda p: p[0])
            
            # 3. Update best
            # (best_x, best_f already updated in safe_call)
            
            # 4. Compute weighted mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * points[i][1]
            y_w /= sigma  # This is the standard normal weighted mean
            
            new_mean = mean + sigma * y_w
            
            # 5. Update step size (CSA)
            # h_sigma = (||y_w|| / sqrt(1 - exp(-2/tau_sigma)) / E||N(0,I)|| > 1.4 + 2/(dim+1))
            # tau_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            tau_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            # E||N(0,I)|| approx sqrt(dim) * (1 - 1/(4*dim))
            E_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim))
            
            # h_sigma is 1 if the condition holds, else 0
            # Simplified h_sigma:
            h_sigma = 1.0 if np.linalg.norm(y_w) / E_norm > 1.4 + 2.0 / (dim + 1.0) else 0.0
            
            # p_sigma = (1 - c_sigma) p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # sigma = sigma * exp(c_sigma / tau_sigma * (||p_sigma|| / E_norm - 1))
            sigma = sigma * np.exp(c_sigma / tau_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # 6. Update covariance matrix
            # p_c = (1 - c_c) p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            # h_c = 1 if (||p_c|| / sqrt(1 - (1 - c_c)^(2*gen))) / E||N(0,I)|| < 1.4 + 2/(dim+1)
            # For simplicity, use h_c = 1.0 (standard in many implementations unless strict)
            h_c = 1.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Rank-1 update: outer(y_w, y_w)
            # Rank-mu update: sum w_i * outer(y_i, y_i)
            y_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (points[i][1] - mean) / sigma
                y_mu += w[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * h_c * (np.outer(p_c, p_c) - C) + c_mu * (y_mu - C)) * (1 + c_1 * (h_c - 1) + c_mu * (h_c - 1))
            # Wait, the standard formula is:
            # C <- (1 - c_1 - c_mu + c_1*h_c*(h_c-1) + c_mu*h_c) * C + c_1*(p_c p_c^T - (1-h_c)C) + c_mu * sum(w_i y_i y_i^T)
            # Let's use the simpler, more stable update:
            # C <- (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_c) * C) + c_mu * y_mu
            # Actually, let's stick to the standard:
            # C = (1 + c_1*(h_c - 1) + c_mu*(h_c - 1)) * C + c_1*(p_c @ p_c.T) + c_mu * y_mu
            # This is equivalent if we adjust the scaling.
            # Let's use the explicit formula from Hansen's paper:
            # C <- (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_c) * C) + c_mu * sum(w_i y_i y_i^T)
            
            # Let's re-calculate C properly
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * C) + c_mu * y_mu
            
            # Ensure positive definiteness and symmetry
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            C += 1e-12 * np.eye(dim)
            
            # Update Cholesky factor
            try:
                C_inv_chol = np.linalg.cholesky(np.linalg.inv(C))
            except np.linalg.LinAlgError:
                # If singular, reset C to identity and restart covariance adaptation
                C = np.eye(dim)
                C_inv_chol = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
            
            # Update mean
            mean = new_mean
            
            # Safety: if mean goes out of bounds, project it back?
            # CMA-ES handles boundaries poorly. Clipping x in safe_call is the primary defense.
            # We can also clip mean to help keep the distribution centered in the box.
            mean = np.clip(mean, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
