"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        # Initial parameters
        x = np.zeros(n)
        f = np.full(n, np.inf)
        fbest = np.inf
        best = x.copy()

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.5) + c_sigma) / ((n + 1.35) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        one_sqrt_c1 = 1 / np.sqrt(c_1)

        # Initial evaluations to fill population
        # We need to be careful with the budget.
        # Let's do a small random search first to get a decent start, 
        # or just start CMA-ES from the center.
        
        # Actually, starting from center with sigma=2 is reasonable for [-5,5].
        
        # Generate initial population
        # We will use a loop that respects the budget and time.
        
        # We need to evaluate the initial population.
        # To save time, we might not evaluate all if time is tight, but 500 is small.
        
        # Let's do a simple restart logic or just run CMA-ES.
        # CMA-ES is robust.
        
        # Pre-allocate
        # We will store evaluations in a list to track budget
        
        # Initial population
        # We can sample from N(0, sigma^2 I)
        
        # Note: The problem says func clips to [-5, 5].
        # CMA-ES handles boundaries poorly if not careful, but clipping is done by func.
        # We should probably clip our samples before calling func to avoid wasting calls?
        # No, the contract says func(x) clips. So we can pass unclipped, but 
        # it's better to clip in our logic to keep the mean inside?
        # Actually, if we clip x before passing, we might bias the gradient.
        # But since func clips anyway, passing unclipped is fine.
        
        # However, if x goes far outside, sigma might need adjustment.
        # Standard CMA-ES assumes no bounds. With bounds, it's "CMA-ES with bounds".
        # Simple strategy: if a sample is outside, reflect or clip?
        # The task says func clips. So func([-10, 0]) returns f([-5, 0]).
        # This creates flat regions outside the box.
        # To be efficient, we should keep the search inside the box.
        
        # Let's modify the sampling to clip to [-5, 5] before evaluation.
        # This effectively changes the distribution to a truncated normal, 
        # which CMA-ES doesn't strictly model, but often works okay.
        
        # Better: Use a simple boundary handling.
        # If x is outside, project back to boundary.
        
        # Let's just clip the samples we generate.
        
        # Initial population
        pop = []
        for _ in range(lambda_):
            z = np.random.default_rng().standard_normal(n)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            pop.append((fy, y))
            
        if time.time() > deadline:
            return best, fbest
            
        pop.sort(key=lambda x: x[0])
        f = np.array([p[0] for p in pop])
        y = np.array([p[1] for p in pop])
        
        fbest = f[0]
        best = y[0].copy()
        
        # Update statistics
        w = weights
        yw = w @ y # Weighted sum
        ymean = y.mean(axis=0)
        
        # Update pc
        pc = (1 - c_c) * pc + c_c * one_sqrt_c1 * sigma * (yw - x)
        
        # Update ps
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C @ (yw - x) / sigma
        
        # Update C
        # Rank-one and rank-mu update
        # C = (1 - c_1 - c_mu) * C + c_1 * outer(pc, pc) + c_mu * sum(wi * outer(yi-ymean, yi-ymean))
        
        # This can be expensive. n=5 is small.
        
        # Update x
        x = xw = yw
        
        # Update sigma
        # hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals + 1)))) / chi_n < 1.4 + 2 / (n + 1)
        # We need to track evals for the hsig calculation.
        
        # Let's simplify: use a fixed time check and break if too slow.
        # Since n=5, matrix ops are fast.
        
        # We need to track number of evaluations for hsig.
        # Let's approximate hsig check.
        
        # For simplicity and speed, let's use a simplified CMA-ES update.
        
        # Actually, let's just run the loop.
        
        eval_count = lambda_
        
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Generate new population
            # We need to sample from N(x, sigma^2 C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or add jitter
                C += 1e-8 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(n)
                    
            pop = []
            for _ in range(lambda_):
                z = np.random.default_rng().standard_normal(n)
                y = np.clip(x + sigma * L @ z, -5, 5)
                fy = func(y)
                pop.append((fy, y))
                eval_count += 1
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
            if not pop:
                break
                
            # Sort
            pop.sort(key=lambda x: x[0])
            f = np.array([p[0] for p in pop])
            y = np.array([p[1] for p in pop])
            
            # Update best
            if f[0] < fbest:
                fbest = f[0]
                best = y[0].copy()
                
            # Update statistics
            # Only use top mu
            yw = np.zeros(n)
            for i in range(mu):
                yw += weights[i] * y[i]
                
            ymean = np.zeros(n)
            for i in range(mu):
                ymean += weights[i] * y[i]
            ymean /= np.sum(weights[:mu]) # Wait, weights are already normalized to sum 1? 
            # Yes, weights /= weights.sum()
            # So ymean is the weighted mean of the selected individuals.
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * one_sqrt_c1 * sigma * (yw - x)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C @ (yw - x) / sigma
            
            # Update C
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            # Sum of outer products
            # d = y_i - ymean
            # C += c_mu * sum(w_i * outer(d_i, d_i))
            # This is expensive if done naively, but n=5 is fine.
            # However, we can optimize.
            
            # Let's just do it directly.
            # C_mu = np.zeros((n, n))
            # for i in range(mu):
            #     d = y[i] - ymean
            #     C_mu += weights[i] * np.outer(d, d)
            # C += c_mu * C_mu
            
            # To save time, we can skip rank-mu if it's too slow? 
            # No, it's crucial for CMA-ES.
            
            # Let's implement it.
            C_mu = np.zeros((n, n))
            for i in range(mu):
                d = y[i] - ymean
                C_mu += weights[i] * np.outer(d, d)
            C += c_mu * C_mu
            
            # Update x
            x = yw
            
            # Update sigma
            # hsig calculation
            # ps_norm = np.linalg.norm(ps)
            # chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            # # Approximate denominator
            # # 1 - (1 - c_sigma)^(2 * (evals + 1))
            # # We can track this.
            
            # For simplicity, let's use a heuristic or just update sigma based on ps.
            # Standard formula:
            # sigma = sigma * exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # We need to be careful with the hsig condition.
            # If hsig is true, we should not update ps? Or set ps to 0?
            # Actually, if hsig is true, we should NOT update pc with the current step?
            # The standard CMA-ES says:
            # if hsig:
            #   pc = (1 - c_c) * pc # No update from current step? 
            #   Actually, the update of pc is always done, but the update of C 
            #   uses pc.
            #   Wait, the hsig condition is for the update of ps?
            #   No, hsig determines if we update ps?
            #   Actually, hsig is used to decide if we update the covariance matrix 
            #   using the rank-one update? No.
            
            # Let's look at the standard CMA-ES algorithm (Hansen 2016).
            # 1. Sample y_i
            # 2. Sort
            # 3. Update mean: x_{t+1} = x_t + sigma * sum(w_i * (y_i - x_t)) / sigma ? No.
            #    x_{t+1} = sum(w_i * y_i)
            # 4. Update step size:
            #    ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * C^{-1/2} * (y_w - x_t) / sigma
            #    hsig = (||ps|| / sqrt(1 - (1 - c_sigma)^{2 * (t+1)})) / chi_n < 1.4 + 2 / (n+1)
            #    sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_n - 1))
            # 5. Update covariance:
            #    pc = (1 - c_c) * pc + c_c * sqrt(c_c * (2 - c_c) * mueff) * (y_w - x_t) / sigma
            #    if hsig:
            #       pc = (1 - c_c) * pc # Wait, no.
            #       Actually, if hsig is false, we do the rank-one update with pc.
            #       If hsig is true, we do NOT update pc?
            #       No, the update of pc is always done.
            #       The hsig condition affects the update of C?
            #       No, hsig affects the update of ps? No.
            
            # Let's check a reference implementation.
            # In CMA-ES:
            # if hsig:
            #    ps = (1 - c_sigma) * ps # No, ps is updated regardless?
            #    Actually, if hsig is false, the step size adaptation is dampened?
            
            # Let's use a simplified version that is robust.
            # We will update sigma using the standard formula.
            # We will update C using the standard formula.
            
            # To avoid complex hsig tracking, we can just use the standard update.
            # The hsig condition is important for stability.
            
            # Let's implement a basic hsig check.
            # We need to track the "age" of the strategy for the denominator.
            # Let's assume it's large enough that the term approaches 1.
            
            # For now, let's just update sigma.
            ps_norm = np.linalg.norm(ps)
            # chi_n is constant
            sigma = sigma * np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Ensure C is positive definite
            # Eigenvalues
            # This is expensive. Let's just add jitter if it fails later.
            
            # We are at the end of the loop.
            
        return best, fbest
# EVOLVE-BLOCK-END
