import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard lambda for d=5 is often 4 + 3*ln(d) ~ 8. Let's use 4 + 3*ln(d)
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        hsigma = np.sqrt((1 - np.exp(-2 / (d + 1))) / (d + 1)) + 1
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.random.randn(lam, d)
            
            # Transformation: x + sigma * B * D * y
            # y is (lam, d). We want z_i = B * D * y_i.
            # Since B is orthogonal and D is diagonal, C = B diag(D^2) B^T.
            # The step is sigma * B * D * y.
            # y_scaled = y * D (element-wise multiplication with D for each row)
            # y_rotated = y_scaled @ B.T  (since B is orthogonal, B * v = (B.T * v.T).T? No.)
            # Let's be precise.
            # y_i is a row vector.
            # v_i = D * y_i (element-wise)
            # z_i = B * v_i (matrix vector product)
            # In numpy: v = ys * D[None, :]
            # z = v @ B.T
            
            v = ys * D
            z = v @ B.T
            pop = x + sigma * z
            
            # Clip to box
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fs = np.array([func(p) for p in pop])
            calls += lam
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = pop[idx_min].copy()
                
            # Sort
            sort_idx = np.argsort(fs)
            x_sorted = pop[sort_idx]
            f_sorted = fs[sort_idx]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update step-size control
            # ps = (1 - c_sigma) * ps + c_sigma * hsigma * (C^(-1/2) * (x_new - x)) / sigma
            # C^(-1/2) = D^(-1) * B.T
            step = (x_new - x) / sigma
            step_unrotated = step @ B.T
            step_uncorrelated = step_unrotated / D
            
            ps = (1 - c_sigma) * ps + c_sigma * hsigma * step_uncorrelated
            
            # Update covariance matrix adaptation
            pc = (1 - c_c) * pc + c_c * (x_new - x)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d * (np.linalg.norm(ps) / np.sqrt(d) - hsigma))
            # Note: Some implementations use c_sigma directly in exp. 
            # Standard: sigma = sigma * exp( c_sigma * (||ps||/sqrt(d) - hsigma) ) 
            # Wait, the standard formula is:
            # sigma = sigma * exp( c_sigma / d_sigma * ( ||ps||/sqrt(d) - hsigma ) )
            # where d_sigma is often 1 or related to c_sigma.
            # Actually, the standard CMA-ES paper (Hansen) uses:
            # sigma = sigma * exp( c_sigma * ( ||ps||/sqrt(d) - hsigma ) )
            # Let's stick to the standard simple form.
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(d) - hsigma))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(w_i * (z_i - z_w) @ (z_i - z_w).T)
            # where z_i are the selected steps in the unrotated space?
            # No, the rank-mu update uses the steps in the original space transformed by C^(-1/2).
            # Let's use the formulation with B and D to avoid explicit C update if possible, 
            # or update C directly.
            
            # Direct C update is safer for correctness if we don't manage B/D perfectly.
            # C_new = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) 
            #        + c_mu * sum_{i=1}^mu w_i * ( (x_i - x)/sigma @ B.T / D ) @ ( ... ).T ?
            
            # Let's compute the rank-mu update term in the unrotated space.
            z_selected = []
            for i in range(mu):
                z = (x_sorted[i] - x) / sigma
                z_unrotated = z @ B.T
                z_uncorrelated = z_unrotated / D
                z_selected.append(z_uncorrelated)
            z_selected = np.array(z_selected)
            
            # Weighted mean of z
            z_w = np.sum(weights[:, np.newaxis] * z_selected, axis=0)
            
            # Outer product sum
            diff = z_selected - z_w
            outer_sum = np.sum(weights[:, np.newaxis, np.newaxis] * diff[:, np.newaxis, :] * diff[:, :, np.newaxis], axis=0)
            
            # C update
            # The term c_mu * outer_sum is in the unrotated space.
            # We need to transform it back to the original space: B @ outer_sum @ B.T
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C)
            C_new += c_mu * (B @ outer_sum @ B.T)
            
            # Ensure C is symmetric and positive definite
            C_new = (C_new + C_new.T) / 2
            
            # Eigen-decompose C to get B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C_new)
                # Clip eigenvalues to be positive
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # Fallback if eigen-decomposition fails
                D = np.ones(d)
                B = np.eye(d)
                C_new = np.eye(d)
            
            C = C_new
            x = x_new
            
        return best_x, best_f
