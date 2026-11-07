import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in tests, but task implies instances are seeded externally? 
        # "Instances apply a hidden seeded shift..."
        # The Optimizer is created fresh for each instance. 
        # Using a fixed seed inside Optimizer is fine, or we can use random. 
        # Given "hidden seeded shift", the function itself is deterministic per instance. 
        # The Optimizer's internal randomness should probably be fixed or random. 
        # Let's use a fixed seed to ensure the algorithm behaves consistently. 
        # Actually, if the function is shifted, a fixed random start might be bad if the shift moves the optimum away. 
        # But CMA-ES adapts. 
        # Let's start at 0,0,0,0,0 which is the center of the box.
        
        N = self.dim
        # Population size
        # Standard recommendation: lambda = 4 + floor(3 * ln(N))
        # For N=5, ln(5)=1.6, 3*1.6=4.8, floor=4. lambda = 8.
        # Let's use 11 to be safe and have more info.
        lam = 11 
        
        # Budget constraint
        # We have 500 calls.
        # Each generation uses lam calls.
        # Max generations = 500 // 11 = 45.
        
        # CMA-ES Parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = (np.sum(weights))**2 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (lam + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lam + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / lam) / (lam + 4 + 2 * mu_eff / lam)
        c_1 = 2 / ((lam + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lam + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(N)
        sigma = 1.0
        B = np.eye(N)
        D = 1.0
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # We need to manage the budget carefully.
        # We will run generations until we run out of budget.
        
        while calls < self.budget:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(N)
                # x_new = x + sigma * B @ (D * z)
                x_new = x + sigma * (B @ (D * z))
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)
            
            # Evaluate population
            # We must evaluate them one by one to respect the call count and update best_f
            # But CMA-ES update requires the sorted population.
            # We can store the results and sort after.
            # However, to maximize AOCC, we want to find the best point as early as possible within the generation?
            # The AOCC is calculated on the sequence of calls.
            # If we evaluate x1, x2... x11, the best after call k is min(f(x1)...f(xk)).
            # So the order doesn't change the *final* best of the generation, but it changes the *trajectory* of b_i.
            # If we happen to evaluate a very good point first, b_i drops early.
            # If we evaluate a bad point first, b_i stays high.
            # Since we don't know which is good, random order is fine.
            
            results = []
            for x_cand in pop:
                if calls >= self.budget:
                    break
                f_val = func(x_cand)
                calls += 1
                results.append((f_val, x_cand))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_cand
            
            if not results:
                break
            
            # Sort results by fitness
            results.sort(key=lambda t: t[0])
            
            # Update CMA-ES parameters
            # Only use the top mu individuals
            y_w = np.zeros(N)
            for i in range(min(mu, len(results))):
                y_w += weights[i] * results[i][1]
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (y_w / sigma if sigma > 0 else 0)
            
            # Update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D @ y_w) ?
            # Standard update:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B @ (D @ (y_w / sigma))
            # Wait, the term is usually:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D @ y_w)
            # Let's check the standard CMA-ES algorithm.
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B @ (D @ (y_w / sigma))
            
            if sigma > 0:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * (y_w / sigma)))
            else:
                ps = np.zeros(N)
            
            # Update sigma
            chi_n = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Covariance Matrix C = B @ diag(D^2) @ B.T
            # We don't store C directly, we store B and D.
            # Update rule:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_pc) * c_c * (2 - c_c) * C) 
            #    + c_mu * sum(wi * yi @ yi.T)
            # Where yi = (xi - x_old) / sigma
            
            # We need to compute the rank-1 and rank-mu updates.
            # Since we store B and D, we can compute C = B @ np.diag(D**2) @ B.T
            C = B @ np.diag(D**2) @ B.T
            
            # h_pc
            # h_pc = 1 if ||pc||^2 / ((1 - (1-c_c)^{2gen}) * N * c_c) < (1 + 2/(N+1)) * chi_n^2
            # We don't track gen count exactly for this, but we can approximate or just use a simple check.
            # For simplicity and robustness in a short run, we can skip the h_pc correction or set it to 1.
            # Let's assume h_pc = 1 for now, or compute it properly.
            # Let's compute it properly.
            # We need the age of pc.
            # Let's just use the standard update with h_pc=1 if we don't track generation count for pc age.
            # Actually, we can track generation count.
            
            # Let's refine the update step.
            # C_new = (1 - c_1 - c_mu) * C
            # + c_1 * (pc @ pc.T + (1 - h_pc) * c_c * (2 - c_c) * C)
            # + c_mu * sum_{i=1}^mu w_i * y_i @ y_i.T
            
            # We need y_i vectors.
            y_vectors = []
            for i in range(min(mu, len(results))):
                y_i = (results[i][1] - x) / sigma if sigma > 0 else np.zeros(N)
                y_vectors.append(y_i)
            
            # Calculate h_pc
            # This requires the history of pc. 
            # Given the short budget, we can approximate h_pc = 1.
            h_pc = 1.0
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            
            for i in range(len(y_vectors)):
                C_new += c_mu * weights[i] * np.outer(y_vectors[i], y_vectors[i])
            
            # Eigen decomposition to get new B and D
            # C_new is symmetric
            eigenvalues, eigenvectors = np.linalg.eigh(C_new)
            # Ensure eigenvalues are positive
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
            
            # Update mean
            x = x + sigma * y_w
            
            # Clip x to bounds?
            # The mean should stay within bounds or close.
            # If x goes out of bounds, the next generation will clip.
            # It's better to keep x within bounds to avoid wasting calls on clipped points that are identical.
            x = np.clip(x, -5, 5)
            
            # If sigma becomes very small, we might converge.
            if sigma < 1e-6:
                break

        return best_x, best_f
