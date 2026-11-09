import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For d=5, lambda=4 is a good choice
        lambda_pop = 4
        mu = lambda_pop // 2
        
        # Weights
        w = np.zeros(mu)
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Control parameters
        cs = (1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1)) / (d + 3 + 2 * mueff / lambda_pop)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        cc = (4 + mueff / lambda_pop) / (d + 4 + 3 * mueff / lambda_pop)
        
        # Initialization
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, d) # Start near center
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        calls = 0
        gen = 0
        
        # Helper to call func and check budget
        def eval_func(x_val):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            # Clip to ensure we stay in bounds for the internal state, 
            # even though func does it, this keeps the "mean" x inside the box
            return func(np.clip(x_val, -5, 5))
        
        # Initial evaluation
        fx = eval_func(x)
        
        while calls < budget and time.time() < deadline:
            gen += 1
            # Generate population
            pop_x = []
            pop_f = []
            for i in range(lambda_pop):
                z = rng.standard_normal(d)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)
                
                x_new = x + sigma * (L @ z)
                f_val = eval_func(x_new)
                pop_x.append(x_new)
                pop_f.append(f_val)
                
                if calls >= budget:
                    break
            
            if len(pop_x) < lambda_pop:
                break
                
            # Sort
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update mean
            x_old = x.copy()
            # x = sum(w_i * x_i)
            x = np.zeros(d)
            for i in range(mu):
                x += w[i] * pop_x[i]
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (x - x_old) / sigma
            
            # Update pc
            # hsig calculation
            denom = np.sqrt(1 - (1 - cs)**(2 * gen))
            hsig = np.linalg.norm(ps) / denom / np.sqrt(d) < 1.4 + 2 / (d + 1)
            
            # y1 is the best individual's step
            y1 = (pop_x[0] - x_old) / sigma
            
            pc = (1 - cc) * pc + hsig * cc * (mueff - 1) * w[0] * y1
            
            # Update C
            delta_hsig = 1 + cs * (1 - cs) / c1 if hsig else 1
            
            # Rank-one update part: c1 * (pc pc^T + (delta_hsig - 1) C)
            # Rank-mu update part: cmu * sum(w_i y_i y_i^T)
            
            rank_one = np.outer(pc, pc) + (delta_hsig - 1) * C
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                yi = (pop_x[i] - x_old) / sigma
                rank_mu += w[i] * np.outer(yi, yi)
                
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(cs / dsigma * (||ps|| / dsigma - 1))
            # dsigma is a constant, usually 1.
            # Actually, the standard update is:
            # sigma = sigma * exp(cs / dsigma * (||ps|| / sqrt(1-(1-cs)^(2/gen)) - 1))
            # Wait, the formula is:
            # sigma = sigma * exp(cs * (||ps|| / sqrt(1-(1-cs)^(2/gen)) - 1))
            # This is slightly different. Let's use the log update.
            # ||ps|| / sqrt(1-(1-cs)^(2/gen)) is the "expected" norm if C=I.
            # So we compare current norm to expected norm.
            
            expected_norm = np.sqrt(1 - (1 - cs)**(2 * gen)) * np.sqrt(d)
            # Actually, the term in the exponent is (||ps|| / expected_norm - 1)
            # But the standard formula uses dsigma.
            # Let's stick to the simpler multiplicative update if possible, 
            # or the standard exponential one.
            
            # Standard: sigma = sigma * exp(cs * (||ps|| / sqrt(1-(1-cs)^(2/gen)) - 1))
            # This assumes dsigma=1.
            
            norm_ps = np.linalg.norm(ps)
            # Avoid division by zero if gen is small
            if denom > 1e-10:
                ratio = norm_ps / denom
            else:
                ratio = 0
                
            # The term is (ratio - 1) ? 
            # The expected norm of ps when C=I is sqrt(1-(1-cs)^(2/gen)) * sqrt(d) ? 
            # No, E[||ps||] is roughly sqrt(d) * sqrt(1-(1-cs)^(2/gen)) is not quite right.
            # The standard formula is:
            # sigma = sigma * exp( cs * ( ||ps|| / sqrt(1-(1-cs)^(2/gen)) - 1 ) )
            # This is correct.
            
            sigma = sigma * np.exp(cs * (ratio - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            if sigma < 1e-10:
                break
            if sigma > 100:
                sigma = 100

        return x, fx
