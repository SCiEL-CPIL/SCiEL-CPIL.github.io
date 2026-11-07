"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Standard lambda for dim=5 is 4 + floor(3*log(5)) = 8. 
        # We use a slightly larger lambda to improve exploration in multi-modal landscapes,
        # but we must be careful with the budget.
        # Let's stick to standard lambda for stability, or maybe 10.
        # For dim=5, lambda=8 is standard.
        lam = 8
        mu = lam // 2
        
        # Initial sigma
        # The problem domain is [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # A good initial step size is roughly 1/5 of the domain width or less.
        # Domain width is 10. 10/5 = 2. Let's start with sigma=1.0 or 0.5.
        # BBOB f4 is Büche-Rastrigin, which has many local minima.
        # A larger initial sigma helps jump over barriers.
        sigma = 1.0
        
        # Initial mean
        # Start at the center? Or random?
        # Center is (0,0,0,0,0).
        # Since the shift is hidden, center is a reasonable guess, but random might be better to avoid bias if the shift is large.
        # However, CMA-ES adapts. Let's start at 0.
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Learning rates
        # Standard values
        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cs = (2 + 2 / (dim + 1. / 3.)) / (dim + cc + 3)
        cmu = min(1 - cs, 2 * (2 * sqrt(dim) - 1) / ((dim + 2) ** 2 + 2))
        
        # Step size control
        d_sigma = 1 + 2 * max(0, sqrt((lam - 2) / (dim + 2)) - 1) + cs
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Inverse square root of C
        # We need to update this every generation
        # C = C + (1/cs + 2*cmu) * p_c p_c^T - ... 
        # Actually, the update is:
        # C = (1 - 1/cs - cmu) * C + 1/cs * p_c p_c^T + cmu * sum( w_i * y_i y_i^T )
        
        # Pre-compute weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        def time_left():
            return deadline - time.time()
            
        best_x = m.copy()
        best_f = np.inf
        calls_made = 0
        
        # Initial evaluation
        f_m = func(m)
        calls_made += 1
        best_f = f_m
        
        while calls_made < budget and time_left() > 0.1:
            # Generate offspring
            # Cholesky factorization of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, restart or fix
                # For robustness, just break or reset C
                break
                
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls_made >= budget or time_left() < 0.1:
                    break
                z = rng.standard_normal(dim)
                y = L @ z
                x = m + sigma * y
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                xs.append(x)
                fs.append(f)
                ys.append(y)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(xs) < lam:
                break
                
            # Sort offspring
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            ys = [ys[i] for i in order]
            
            # Update mean
            m_old = m.copy()
            m = m + sum(w[i] * (xs[i] - m_old) for i in range(mu))
            
            # Update step size
            y_sigma = (L.T @ p_sigma) / sigma # This is not quite right, need to be careful with the path update
            # Actually, p_sigma update is:
            # p_sigma = (1 - d_sigma) * p_sigma + sqrt(d_sigma * (2 - d_sigma) * mu_eff) * (1/sigma) * L.T @ (m - m_old)
            
            # Let's re-implement the update steps carefully.
            
            # 1. Update mean (done above, but we used m_old which is correct)
            
            # 2. Update sigma
            # p_sigma = (1 - d_sigma) * p_sigma + sqrt(d_sigma * (2 - d_sigma) * mu_eff) * (1/sigma) * L.T @ (m - m_old)
            # Note: L is from the previous iteration's C.
            # (m - m_old) is the actual step.
            # We need L.T @ (m - m_old).
            # Let v = m - m_old
            v = m - m_old
            p_sigma = (1 - d_sigma) * p_sigma + np.sqrt(d_sigma * (2 - d_sigma) * mu_eff) / sigma * (L.T @ v)
            
            # Heuristic check for delta of normal
            delta_n = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - d_sigma) ** (2 * (calls_made - dim - 1) / lam))) - np.sqrt(dim - 1) * (1 + 1 / (dim + 2))
            
            sigma = sigma * np.exp(cs / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # 3. Update C
            # p_c = (1 - cs) * p_c + sqrt(cs * (2 - cs) * mu_eff) * (1/sigma) * (m - m_old)
            # Wait, the formula for p_c uses the normalized step?
            # p_c = (1 - cs) * p_c + sqrt(cs * (2 - cs) * mu_eff) * (1/sigma) * (m - m_old)
            # Actually, the standard CMA-ES update for p_c is:
            # p_c = (1 - cs) * p_c + sqrt(cs * (2 - cs) * mu_eff) * (1/sigma) * (m - m_old)
            # But (m - m_old) is already in "sigma" units? No, it's in x-space.
            # The step in y-space is (m - m_old) / sigma.
            # So p_c = (1 - cs) * p_c + sqrt(cs * (2 - cs) * mu_eff) * ((m - m_old) / sigma)
            
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (v / sigma)
            
            # h_sigma = (np.linalg.norm(p_sigma) / sqrt(1 - (1 - d_sigma) ** (2 * lam / (dim + 2))) / np.sqrt(dim)) < (1 + 2 / (dim + 2)) * c_sigma
            # This condition is complex. Let's use a simpler check or skip h_sigma if we are not implementing the full robust version.
            # For dim=5 and small budget, a simplified CMA-ES might suffice.
            # Let's assume h_sigma = 1 for now to simplify, or calculate it properly.
            # Let's calculate h_sigma properly.
            # The term (1 - d_sigma) ** (2 * lam / (dim + 2)) is the decay factor for the path length.
            # Actually, the standard check is:
            # h_sigma = 1 if norm(p_sigma) / sqrt(1 - (1-d_sigma)^(2 * t_gen * lam / (dim+2))) / sqrt(dim) < (1 + 2/(dim+2)) * c_sigma
            # Since we don't track t_gen exactly in the loop counter this way, let's approximate.
            # Or just use h_sigma = 1.0 if we want to be safe and simple.
            # Let's try to compute it.
            # t_gen is roughly calls_made / lam.
            # Let's just use h_sigma = 1.0 for simplicity in this constrained budget.
            h_sigma = 1.0
            
            # C update
            # C = (1 - 1/cs - cmu) * C + 1/cs * p_c p_c^T + cmu * sum( w_i * y_i y_i^T )
            # Note: y_i are the selected y's (before sigma scaling? Yes, y = (x - m_old)/sigma)
            # Our ys list contains the y's used to generate xs.
            # We need the y's of the selected individuals.
            # xs[i] = m_old + sigma * ys[i]
            # So (xs[i] - m_old) / sigma = ys[i]
            
            C = (1 - 1.0 / cs - cmu) * C + (1.0 / cs) * np.outer(p_c, p_c)
            for i in range(mu):
                C += cmu * w[i] * np.outer(ys[i], ys[i])
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small diagonal if needed
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)

        return best_x, best_f

# EVOLVE-BLOCK-END
