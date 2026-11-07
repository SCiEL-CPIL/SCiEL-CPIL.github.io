import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # CMA-ES Parameters
        # Lambda (population size)
        lambda_ = 4 + int(np.floor(np.log(dim)))
        if lambda_ < 2:
            lambda_ = 2
        # For dim=5, lambda = 4 + 1 = 5. Let's use 9 for better exploration? 
        # Standard heuristic is 4 + floor(ln(n)). For n=5, ln(5)=1.6, floor=1. Lambda=5.
        # However, small lambda can be noisy. Let's try lambda=9.
        # Actually, for small budgets, larger lambda might waste calls.
        # Let's stick to a slightly larger lambda for stability: lambda = 9.
        lambda_ = 9
        
        mu = lambda_ // 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES Strategy Parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (mu_eff + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (32 * dim**2))
        
        # Initialization
        # Start near 0, which is often a good heuristic for BBOB instances if shift is not too large
        # Or random. Let's start at 0.
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers [-2, 2] roughly
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for sampling
        # C = B @ diag(D^2) @ B.T
        # Initially C=I, so B=I, D=1
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation of mean
        f0 = eval_func(x_mean)
        
        # Main Loop
        while calls < budget and time.time() < self.deadline:
            # Generate population
            pop = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Sample y
                y = rng.standard_normal(dim)
                # Transform
                # x_i = x_mean + sigma * B @ (D * y)
                # Note: B @ D is the square root of C
                x_i = x_mean + sigma * (B @ (D * y))
                pop[i] = x_i
                fvals[i] = eval_func(x_i)
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort population
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            # (Already handled in eval_func)
            
            # Check for time/budget
            if time.time() > self.deadline:
                break
            
            # Update Mean
            x_old = x_mean.copy()
            x_mean = np.zeros(dim)
            for i in range(mu):
                x_mean += w[i] * pop[i]
            
            # Update Step Size (p_sigma)
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * B @ D^-1 (x_mean - x_old)
            # Note: B @ D^-1 is the inverse of the square root of C
            # Since C = B D^2 B^T, C^-1/2 = B D^-1 B^T
            # The vector (x_mean - x_old) is in the transformed space?
            # Standard formula: p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} (x_mean - x_old) / sigma
            # C^{-1/2} = B @ diag(1/D) @ B.T
            diff = x_mean - x_old
            # Compute C^{-1/2} @ diff
            # diff_inv = B @ ( (1/D) * (B.T @ diff) )
            diff_inv = B @ ((1.0 / D) * (B.T @ diff))
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff_inv / sigma
            
            # Update sigma
            # sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/chi_n - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance Path (p_c)
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} (x_mean - x_old) / sigma
            # Note: The term C^{-1/2} (x_mean - x_old) / sigma is the same as diff_inv / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_inv / sigma
            
            # Update Covariance Matrix C
            # Rank-1 update
            # C = (1 + c_1 (mu_eff - 2) - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_sigma (2 - c_sigma) C)
            # h_sigma = 1 if ||p_sigma||^2 / (1 - (1-c_sigma)^{2gen}) < (1.4 + 2/(dim+1)) chi_n^2
            # For simplicity and robustness in small runs, we can approximate h_sigma
            # Or just use the standard condition.
            # Let's implement the standard h_sigma check roughly.
            # We need to track generation or use a heuristic.
            # Given small budget, let's assume h_sigma = 1 if norm is large, else 0.
            # Actually, a simpler robust version often used:
            # h_sigma = 1 if (norm_p_sigma**2) / (1 - (1-c_sigma)**(2*gen)) < (1.4 + 2/(dim+1)) * chi_n**2 else 0
            # We don't have 'gen' explicitly, but we can count iterations.
            # Let's add a generation counter.
            
            # Rank-mu update
            # + c_mu sum_{i=1}^mu w_i y_i^+ y_i^{+T}
            # where y_i^+ = C^{-1/2} (x_i - x_old) / sigma
            
            # We need C^{-1/2} for all top mu points
            # y_plus = np.zeros((mu, dim))
            # for i in range(mu):
            #     diff_i = pop[i] - x_old
            #     y_plus[i] = B @ ((1.0 / D) * (B.T @ diff_i)) / sigma
            
            # Let's compute the update term
            # Term1 = (1 + c_1 * (mu_eff - 2) - c_1 - c_mu) * C
            # Term2 = c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_sigma * (2 - c_sigma) * C)
            # Term3 = c_mu * sum(w_i * outer(y_plus_i, y_plus_i))
            
            # To save computation, we can update C directly.
            # However, B and D need to be updated via eigendecomposition.
            
            # Let's do the eigendecomposition every generation since dim=5 is small.
            
            # First, compute y_plus for rank-mu update
            y_plus = np.zeros((mu, dim))
            for i in range(mu):
                diff_i = pop[i] - x_old
                # C^{-1/2} diff_i / sigma
                y_plus[i] = (B @ ((1.0 / D) * (B.T @ diff_i))) / sigma
            
            # Estimate h_sigma
            # We need a generation counter. Let's use a local variable.
            # We can approximate the denominator as 1 for large gen, or track it.
            # For simplicity, let's use a heuristic: if norm_p_sigma is close to chi_n, h=1, else 0.
            # More accurately:
            # threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n**2
            # We need the cumulative sum of (1-c_sigma)^2.
            # Let's just use h_sigma = 1 if norm_p_sigma < 1.5 * chi_n else 0.
            h_sigma = 1 if norm_p_sigma < 1.5 * chi_n else 0
            
            # Compute C update
            # C_new = (1 - c_1 - c_mu) * C + c_1 * (outer(p_c, p_c) + (1-h_sigma)*c_sigma*(2-c_sigma)*C) + c_mu * sum(...)
            
            # Note: The coefficient for C in the rank-1 part is c_1 * (1 - h_sigma) * c_sigma * (2 - c_sigma)
            # And the base decay is (1 - c_1 - c_mu)
            
            C_sum = np.zeros((dim, dim))
            for i in range(mu):
                C_sum += w[i] * np.outer(y_plus[i], y_plus[i])
                
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * np.outer(p_c, p_c)
            C_new += c_1 * (1 - h_sigma) * c_sigma * (2 - c_sigma) * C
            C_new += c_mu * C_sum
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Eigen decomposition
            # C_new = B_new @ diag(D_new^2) @ B_new.T
            eigenvalues, B_new = np.linalg.eigh(C_new)
            # Clip small negative eigenvalues
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            D_new = np.sqrt(eigenvalues)
            
            C = C_new
            B = B_new
            D = D_new
            
            # Check for convergence or stagnation
            # If sigma is very small or C is singular, restart?
            # With 500 budget, restarts might be expensive.
            # If no improvement for many generations, break?
            # The eval_func tracks best_f. We can check if best_f hasn't changed.
            # For now, just run until budget/time.
            
        return best_x, best_f
