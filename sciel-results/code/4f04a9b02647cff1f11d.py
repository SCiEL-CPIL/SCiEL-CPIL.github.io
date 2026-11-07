import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-Start Exploration
        # Evaluate a small number of random points to find a good starting region.
        # This handles the multimodality of Büche-Rastrigin better than a single start.
        n_starts = 10
        if budget < n_starts + 10:
            n_starts = max(1, budget // 10)
        
        candidates = rng.uniform(-5, 5, (n_starts, n))
        candidate_f = []
        calls = 0
        
        for c in candidates:
            if time.time() > deadline or calls >= budget:
                break
            fc = func(c)
            candidate_f.append(fc)
            calls += 1
            
        if not candidate_f:
            # Fallback
            x = np.zeros(n)
            best_f = func(x)
            calls += 1
            best_x = x.copy()
            return best_x, best_f
            
        best_idx = np.argmin(candidate_f)
        x = candidates[best_idx].copy()
        best_f = candidate_f[best_idx]
        best_x = x.copy()
        
        # Phase 2: CMA-ES Refinement
        # Use CMA-ES for the remaining budget to exploit the local landscape.
        # CMA-ES adapts the covariance matrix, which is powerful for rotated landscapes.
        
        lam = 4 + int(3 * np.log(n))
        if lam > budget - calls:
            lam = max(2, budget - calls)
            
        w = np.log(lam/2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mu = int(np.sum(w > 0))
        w = w[:mu]
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        L = np.linalg.cholesky(C)
        sigma = 1.0  # Start with a moderate step size
        
        # Generate initial population around the best start
        samples = []
        fvals = []
        for i in range(lam):
            if time.time() > deadline or calls >= budget:
                break
            z = rng.standard_normal(n)
            s = x + sigma * L @ z
            s_clipped = np.clip(s, -5, 5)
            samples.append(s_clipped)
            fs = func(s_clipped)
            fvals.append(fs)
            calls += 1
            if fs < best_f:
                best_f = fs
                best_x = s_clipped.copy()
                
        if not samples:
            return best_x, best_f
            
        # Main CMA-ES Loop
        while calls < budget and time.time() < deadline:
            order = np.argsort(fvals)
            fvals_sorted = np.array(fvals)[order]
            samples_sorted = [samples[i] for i in order]
            
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = samples_sorted[0].copy()
                
            x_old = x.copy()
            # Update mean
            x = sum(w[i] * samples_sorted[i] for i in range(min(mu, len(samples_sorted))))
            
            d = x - x_old
            norm_d = np.linalg.norm(d)
            if norm_d > 0:
                d = d / norm_d
            else:
                d = np.zeros(n)
                
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * d
            
            gen = calls / lam
            denom = np.sqrt(1 - (1 - cc) ** gen)
            hsig = (np.linalg.norm(pc) / denom) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cc) ** (2 * gen))
            
            if hsig:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (np.linalg.solve(L, x - x_old) / sigma)
            else:
                ps = (1 - cs) * ps
            
            rank1 = np.outer(pc, pc)
            rankmu = sum(w[i] * np.outer(samples_sorted[i] - x_old, samples_sorted[i] - x_old) for i in range(min(mu, len(samples_sorted))))
            
            C = (1 + c1 * (hsig - 1) + cmu - c1 - cmu) * C \
                + c1 * (rank1 + (hsig - 1) * np.eye(n)) \
                + cmu * rankmu
            
            C = (C + C.T) / 2
            
            sigma = sigma * np.exp(cs / (2 * (n + 1)) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** gen) - np.sqrt(n)))
            
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(n) * (np.trace(C) / n)
                L = np.linalg.cholesky(C)
                
            # Generate next population
            new_samples = []
            new_fvals = []
            for i in range(lam):
                if time.time() > deadline or calls >= budget:
                    break
                z = rng.standard_normal(n)
                s = x + sigma * L @ z
                s_clipped = np.clip(s, -5, 5)
                new_samples.append(s_clipped)
                fs = func(s_clipped)
                new_fvals.append(fs)
                calls += 1
                if fs < best_f:
                    best_f = fs
                    best_x = s_clipped.copy()
                    
            if not new_samples:
                break
                
            samples = new_samples
            fvals = new_fvals
            
        return best_x, best_f
