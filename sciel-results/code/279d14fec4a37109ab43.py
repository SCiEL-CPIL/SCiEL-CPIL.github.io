"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        n = self.dim
        budget = self.budget
        calls = 0
        
        def time_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        # Standard recommendations for n=5
        lambda_ = 4 + int(np.floor(3 * np.log(n)))  # Population size
        mu = lambda_ // 2  # Number of weighted individuals
        # Weights: linear rank weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Step size sigma
        sigma = 2.0  # Initial step size, covers the box [-5,5]^5 reasonably
        
        # Initial mean
        m = np.zeros(n)
        
        # Initial covariance matrix
        C = np.eye(n)
        
        # Precompute Cholesky decomposition of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(n)
            
        # Precompute D (square root of trace(C))
        D = np.sqrt(np.trace(C))
        
        # CMA-ES update parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m.copy()
        best_f = np.inf
        
        # Main loop
        # We need to be careful with the budget. 
        # Each generation uses lambda_ calls.
        # We will run as many generations as possible within the budget.
        
        while calls < budget and time_ok():
            # Check if we have enough budget for a full generation
            if calls + lambda_ > budget:
                # If not, do a final small random search or just break
                # To be safe and simple, break if we can't do a full gen
                # Or do partial? CMA-ES is sensitive to population size.
                # Let's just break to avoid incomplete stats.
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                if not time_ok():
                    break
                y = rng.standard_normal(n)
                z = B @ y
                x = m + sigma * z
                # Clip to box? The problem says func clips, but CMA-ES prefers unconstrained.
                # However, if we go far out, we might waste calls. 
                # The problem states: "func(x) clips a finite length-5 vector to the box".
                # So we can pass x directly.
                fx = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(fs) < lambda_:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            m_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # Calculate p_sigma and p_c
            y_sigma = (m_new - m) / (sigma * D)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian update approximation for p_c
            # p_c = (1 - c_c) * p_c + c_c * ( ||p_sigma|| < (1.4 + 2/(n+1)) * sqrt((1-c_sigma)/(1-c_c)) ) * (m_new - m)/D
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Check condition for p_c update
            # Note: We need to be careful with the threshold
            if np.linalg.norm(p_sigma) < (1.4 + 2/(n+1)) * np.sqrt((1-c_sigma)/(1-c_c)):
                p_c = (1 - c_c) * p_c + c_c * (m_new - m) / D
            else:
                p_c = (1 - c_c) * p_c
            
            # Update C
            # h_sigma = 1 if ||p_sigma||^2 / (1 - (1-c_sigma)^{2*(gen+1)}) < (2*chi_n+0.3)*chi_n else 0
            # We don't track 'gen' explicitly in a way that's easy here without more state, 
            # but for a short run, we can approximate or skip the h_sigma check 
            # or use a simple heuristic. 
            # Actually, let's implement a basic h_sigma check using a stored variable.
            # For simplicity and robustness in a black-box context with limited budget,
            # we can often get away with just the rank-one and rank-mu updates 
            # without the strict h_sigma condition if we are careful, 
            # but the condition prevents divergence.
            
            # Let's assume h_sigma = 1 for now, or calculate it roughly.
            # A safer bet for a short run:
            h_sigma = 1.0 # Simplification
            
            # Rank-one update
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 * (p_c + h_sigma * c_sigma * p_sigma) @ (p_c + h_sigma * c_sigma * p_sigma).T
            
            # Rank-mu update
            ys_normed = ys_sorted[:mu] / (sigma * D)
            C += c_mu * np.sum(w[:, None, None] * np.einsum('i,ij,ik->jk', ys_normed, ys_normed, ys_normed), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Re-estimate Cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If it becomes singular, add small diagonal
                C += 1e-6 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                    
            D = np.sqrt(np.trace(C))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            m = m_new
            
            # If we are very close to the minimum, we might want to stop or reduce sigma
            # But CMA-ES handles this via sigma adaptation.

        return best_x, best_f
# EVOLVE-BLOCK-END
