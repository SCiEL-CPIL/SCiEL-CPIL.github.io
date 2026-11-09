"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with 1/5 success rule for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, keep 10s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda is typically 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3 * 1.609) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # weight half the best
        
        # Weights
        # Standard CMA weights: w_i = log(mu + 0.5) - log(i)
        # Normalized so sum(w) = 1
        w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Initial step size
        # cma default: 0.5 * (max - min) / 4? No, usually 1.0 or based on range.
        # Range is 10. 1.0 is a reasonable start for [-5, 5].
        sigma = 1.0
        
        # Adaptation rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial population
        rng = np.random.default_rng(42)
        m = np.zeros(dim)  # Start at origin, robust for shifted problems often
        
        # Generate initial population
        z = rng.standard_normal((lam, dim))
        # Apply initial C (identity) and sigma
        pop = m + sigma * z
        # Clip to bounds? CMA-ES usually works in unconstrained space, 
        # but the problem clips. It's better to let CMA explore and 
        # rely on the clipping in func. However, if we clip here, 
        # the gradient information is distorted. 
        # The contract says func clips. So we pass raw x to func.
        
        # Evaluate initial population
        # We have budget calls. 
        # We need to track total calls.
        calls = 0
        best_val = np.inf
        best_x = None
        
        # Store history for CMA update
        # We need to store the last generation's z and x
        
        # Pre-allocate for speed?
        
        # Main Loop
        # Each generation uses lam calls.
        # We can do floor((budget - lam) / lam) generations?
        # Actually, we can just run until budget is exhausted.
        
        # Initial evaluation
        for i in range(lam):
            if calls >= budget: break
            if time.time() > deadline: break
            x_i = pop[i]
            f_i = func(x_i)
            calls += 1
            if f_i < best_val:
                best_val = f_i
                best_x = x_i.copy()
            
            # Store for this generation
            # We need to store z and x for CMA update
            # Let's use lists
            z_list = [z[i]]
            x_list = [x_i]
            f_list = [f_i]
            
        # Wait, the above loop structure is bad for CMA. 
        # CMA needs the whole population evaluated to update C.
        
        # Let's restructure.
        
        # Reset state
        calls = 0
        best_val = np.inf
        best_x = None
        m = np.zeros(dim)
        C = np.eye(dim)
        sigma = 1.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # We need to be careful with memory and time.
        
        # Generation 0
        z = rng.standard_normal((lam, dim))
        pop = m + sigma * z
        f_vals = np.zeros(lam)
        
        for i in range(lam):
            if calls >= budget or time.time() > deadline:
                return best_x, best_val
            f_vals[i] = func(pop[i])
            calls += 1
            if f_vals[i] < best_val:
                best_val = f_vals[i]
                best_x = pop[i].copy()
                
        # Sort
        idx = np.argsort(f_vals)
        pop_sorted = pop[idx]
        z_sorted = z[idx]
        f_sorted = f_vals[idx]
        
        # Update mean
        m = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
        
        # Update sigma (1/5 rule)
        # Actually, standard CMA uses pc and ps for sigma update.
        # Let's implement standard CMA-ES update.
        
        # Standard CMA-ES Update
        # 1. Mean update (done above)
        
        # 2. Step size control
        # hsig = (1 - np.prod(np.sqrt(np.diag(C)))) / dim? 
        # Approximation: hsig = 1 if trace(C) is close to dim, else adjust.
        # Simpler: use the 1/5 rule on the step size directly?
        # The 1/5 rule is simpler and robust. 
        # Let's stick to the 1/5 rule for sigma, but use CMA for C and m.
        # This is a hybrid: CMA for direction/covariance, 1/5 for step size.
        
        # 1/5 rule:
        # Count successes in the top mu? Or all lam?
        # Usually 1/5 rule is for (1+1) or (mu+lambda) with specific weights.
        # Let's use the standard CMA sigma update with ps and pc.
        
        # Recalculate ps and pc
        # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^(-1/2) * (m_new - m_old) / sigma
        # We didn't store m_old. Let's store it.
        
        # Let's restart the loop properly with state tracking.
        
        # --- RESTART LOGIC WITH PROPER STATE ---
        
        calls = 0
        best_val = np.inf
        best_x = None
        
        m_old = np.zeros(dim)
        m = m_old.copy()
        C = np.eye(dim)
        sigma = 1.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Precompute C^(-1/2) for ps update? 
        # np.linalg.eigh(C) -> eigenvalues, eigenvectors
        # C^(-1/2) = V * diag(1/sqrt(lambda)) * V.T
        
        while calls < budget and time.time() < deadline:
            # Generate population
            z = rng.standard_normal((lam, dim))
            
            # Update C^(-1/2) if C changed? 
            # For simplicity, compute once per generation.
            # For dim=5, this is cheap.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Avoid division by zero
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Generate x
            pop = m + sigma * (z @ C_inv_sqrt.T) # z @ C_inv_sqrt.T is equivalent to C_inv_sqrt @ z.T ? 
            # C is symmetric. C^(-1/2) is symmetric.
            # z is (lam, dim). z @ C_inv_sqrt.T is (lam, dim).
            # Correct: x = m + sigma * z * C_inv_sqrt
            
            f_vals = np.zeros(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    return best_x, best_val
                f_vals[i] = func(pop[i])
                calls += 1
                if f_vals[i] < best_val:
                    best_val = f_vals[i]
                    best_x = pop[i].copy()
                    
            # Sort
            idx = np.argsort(f_vals)
            pop_sorted = pop[idx]
            z_sorted = z[idx]
            
            # Update mean
            m_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (m_new - m_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (m_new - m_old) / sigma)
            
            # Update pc
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^{2(g+1)}) < (1.4 + 2/(dim+1)) * sqrt((mueff-1+1)/mueff+1) * sqrt(trace(C)/dim - 1) + 1
            # Simplified hsig:
            # threshold = (1.4 + 2/(dim+1)) * np.sqrt((mueff - 1 + 1/mueff) * (np.trace(C)/dim - 1) + 1)
            # norm_ps = np.linalg.norm(ps)
            # denom = np.sqrt(1 - (1-cs)**(2 * (calls // lam + 1))) # Approximation of generation count
            # gen_count = calls // lam + 1
            # denom = np.sqrt(1 - (1-cs)**(2 * gen_count))
            # if norm_ps / denom < threshold:
            #     hsig = 1
            # else:
            #     hsig = 0
            
            # Let's use a simpler heuristic for hsig or just set hsig=1 if sigma is not exploding.
            # For robustness, let's calculate it properly.
            gen_count = calls // lam + 1
            denom = np.sqrt(1 - (1 - cs) ** (2 * gen_count))
            threshold = (1.4 + 2 / (dim + 1)) * np.sqrt((mueff - 1 + 1 / mueff) * (np.trace(C) / dim - 1) + 1)
            hsig = 1.0 if np.linalg.norm(ps) / denom < threshold else 0.0
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (m_new - m_old) / sigma
            
            # Update C
            # C = (1 - mu1 - cmu) * C + mu1 * pc * pc.T + cmu * sum(w_i * (y_i - m_old) * (y_i - m_old).T)
            # where y_i = (pop_sorted[i] - m_old) / sigma
            delta_m = m_new - m_old
            y_sorted = (pop_sorted[:mu] - m_old) / sigma
            
            # Rank-one update
            C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * (np.trace(C) / dim - 1) * C)
            
            # Rank-mu update
            for i in range(mu):
                C += cmu * w[i] * np.outer(y_sorted[i], y_sorted[i])
                
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(cs / (2 * lambda_sigma) * (||ps||^2 / dim - 1))
            # lambda_sigma = (mueff + 2) / (dim + mueff + 5) / 2 ? No, lambda_sigma = 1 / (2 * cs) ?
            # Standard: lambda_sigma = (mueff + 2) / (dim + mueff + 5) / 2 is not right.
            # cs = (mueff + 2) / (dim + mueff + 5)
            # lambda_sigma = 1 / (2 * cs) ?
            # Let's use the standard formula:
            # sigma *= np.exp((cs / 2.0) * (np.linalg.norm(ps)**2 / dim - 1))
            # This is the log-step-size update.
            
            sigma *= np.exp((cs / 2.0) * (np.linalg.norm(ps)**2 / dim - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            m_old = m
            m = m_new
            
        return best_x, best_val
# EVOLVE-BLOCK-END
