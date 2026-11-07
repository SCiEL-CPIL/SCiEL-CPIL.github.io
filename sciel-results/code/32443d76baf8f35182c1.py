"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        evals_used = 0
        
        # Helper to safely call func and update best
        def safe_call(x):
            nonlocal best_x, best_f, evals_used
            if time.time() > deadline:
                return best_f
            try:
                f_val = func(x)
                evals_used += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return float('inf')
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                # If an exception occurs, we assume it's a bad point or stop.
                # The prompt says "exception later keeps the calls made".
                # But we must not crash.
                return float('inf')

        # 1. Initial Sampling: 5 random points in [-4, 4]
        # This helps avoid starting in a bad local minimum for multimodal functions.
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(-4, 4, dim)
            safe_call(x)
            if evals_used >= budget or time.time() > deadline:
                break

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # 2. CMA-ES Optimization
        # Population size lambda = 4 + floor(3 * ln(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        # Ensure lam is not too large for the remaining budget
        max_lam = max(4, (budget - evals_used) // 5) # Allow at least 5 generations
        lam = min(lam, max_lam)
        
        # CMA-ES Parameters
        sigma = 1.0
        # Initialize mean at the best found so far
        mean = best_x.copy()
        
        # Covariance matrix C
        C = np.eye(dim)
        # Step sizes
        c_sigma = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((lam - 1) / (dim + 1) - 1) - 1)
        c_c = (4 + lam / dim) / (dim + 9)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 - 1) / (dim + 2) - 1))
        
        # Weights
        w = np.log(np.arange(1, lam / 2 + 1)) - np.log(lam / 2)
        w = w / w.sum()
        mu_eff = 1 / np.sum(w[:lam // 2] ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Main Loop
        while evals_used < budget and time.time() < deadline:
            # Generate population
            # If remaining budget is less than lam, we might not complete a generation.
            # We stop if we can't complete a full generation or if time is up.
            if budget - evals_used < lam:
                break
                
            # Sample weights
            weights = w.copy()
            # Standard CMA-ES sampling
            # p = mean + sigma * sum(w_i * z_i * C^0.5)
            # We use the standard update rule.
            
            # Generate random normals
            Z = rng.standard_normal((lam, dim))
            # Transform Z using C^0.5 (Cholesky or Eigen)
            # For 5D, Cholesky is fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                L = np.eye(dim)
            
            # B = L * D^0.5 where D is diagonal of eigenvalues? No, standard CMA-ES uses C directly in the update.
            # Actually, the standard implementation samples y = C^0.5 * z.
            # We can compute C^0.5 via eigen decomposition or Cholesky.
            # Let's use eigen for stability in 5D.
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 0, None)
            sqrt_vals = np.sqrt(vals)
            C_sqrt = vecs @ np.diag(sqrt_vals) @ vecs.T
            
            Y = Z @ C_sqrt.T # (lam, dim)
            
            # Generate candidates
            X = mean[None, :] + sigma * Y
            # Clip to bounds [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = []
            for i in range(lam):
                f_val = safe_call(X[i])
                F.append(f_val)
                if time.time() > deadline or evals_used >= budget:
                    break
            
            if len(F) < lam:
                break
                
            F = np.array(F)
            
            # Sort indices by fitness
            idx = np.argsort(F)
            # Select top mu_eff individuals
            # Standard CMA-ES uses the best lam/2 individuals weighted.
            # Here we use the top lam//2.
            selected_idx = idx[:lam // 2]
            
            # Update mean
            # m_new = sum(w_i * x_i)
            # Note: w is defined for the first lam//2 individuals in sorted order.
            m_new = np.sum(weights[:, None] * X[selected_idx], axis=0)
            
            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C^(-0.5) * (m_new - mean)
            # We need C^(-0.5).
            try:
                C_inv_sqrt = vecs @ np.diag(1 / np.sqrt(vals + 1e-10)) @ vecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < (1.4 + 2 / (dim + 1))
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * (m_new - mean) / sigma
            else:
                pc = (1 - c_c) * pc
                
            # Update C
            # delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            # Rank-one update
            # C += (c_1 + c_mu) * (y * y.T + delta_hsig * pc * pc.T)
            y = (m_new - mean) / sigma
            C = (1 + c_sigma * (c_1 + c_mu - 1) + c_1 * delta_hsig) * C \
                + c_1 * (y[:, None] * y[None, :] + delta_hsig * pc[:, None] * pc[None, :])
            
            # Rank-mu update
            # C += c_mu * sum(w_i * y_i * y_i.T)
            Y_sel = X[selected_idx] - mean
            Y_sel = Y_sel / sigma
            C += c_mu * np.sum(weights[:, None, None] * Y_sel[:, :, None] * Y_sel[:, None, :], axis=0)
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            # Clip small negative eigenvalues
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-10, None)
            C = vecs @ np.diag(vals) @ vecs.T
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = m_new
            
            # Convergence check
            if np.linalg.norm(sigma * np.sqrt(np.trace(C)) / dim) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
