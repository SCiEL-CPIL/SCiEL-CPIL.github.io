import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        rng = np.random.default_rng(42)
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
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
        
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        L = np.linalg.cholesky(C)
        samples = []
        for i in range(lam):
            if time.time() > deadline:
                break
            z = rng.standard_normal(n)
            s = x + sigma * L @ z
            s_clipped = np.clip(s, -5, 5)
            samples.append(s_clipped)
            calls += 1
            
        if calls == 0:
            return best_x, best_f

        fvals = []
        for s in samples:
            if time.time() > deadline:
                break
            fvals.append(func(s))
            
        if len(fvals) < len(samples):
            samples = samples[:len(fvals)]
            lam = len(samples)
            # Recalculate w, mu, mueff if lam changed? No, keep original params, just use available data.
            # This is an edge case, unlikely to happen.
            
        while calls < self.budget and time.time() < deadline:
            order = np.argsort(fvals)
            fvals = np.array(fvals)
            fvals_sorted = fvals[order]
            samples_sorted = [samples[i] for i in order]
            
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = samples_sorted[0]
                
            x_old = x.copy()
            # Use only top mu samples for mean update
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
                
            new_samples = []
            new_fvals = []
            for i in range(lam):
                if calls >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                s = x + sigma * L @ z
                s_clipped = np.clip(s, -5, 5)
                new_samples.append(s_clipped)
                calls += 1
                new_fvals.append(func(s_clipped))
                
            if not new_samples:
                break
                
            samples = new_samples
            fvals = new_fvals
            
        return best_x, best_f
