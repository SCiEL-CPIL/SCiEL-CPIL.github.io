import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        n = self.dim
        
        # --- Phase 1: CMA-ES ---
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.dot(w, w)
        
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        Cs = np.zeros((n, n))
        p_sigma = np.zeros(n)
        
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 - mueff) / (n + 4)
        cmu = min(1 - cc, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (mueff + 5)))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        cma_budget = int(0.6 * self.budget)
        
        while calls < cma_budget:
            if time.time() > deadline:
                break
                
            # Sample
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            population = np.array([x_mean + sigma * sqrtC @ rng.standard_normal(n) for _ in range(lam)])
            fvals = np.array([func(x) for x in population])
            calls += lam
            
            # Sort
            order = np.argsort(fvals)
            y_selected = (population[order[:mu]] - x_mean) / sigma
            
            # Update mean
            x_mean_new = x_mean + sigma * np.dot(w, y_selected)
            
            # Update evolution path
            y_mean = np.dot(w, y_selected)
            Cs = (1 - cs) * Cs + cs * (2 - cs) * (y_mean[:, None] * y_mean[None, :] - C)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            
            # Update C
            delta_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls / lam))) - 1) * (cc + 1)
            C = (1 - cs * (2 - cs) - cmu) * C + cs * (2 - cs) * delta_sigma * Cs + cmu * np.dot(w, y_selected[:, None] * y_selected[None, :])
            
            C = (C + C.T) / 2
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.clip(eigvals_C, 1e-12, None)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
            
            x_mean = x_mean_new
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (calls / lam)) - 1))
            
            # Track best
            if fvals[order[0]] < best_f:
                best_f = fvals[order[0]]
                best_x = population[order[0]]

        # --- Phase 2: Nelder-Mead ---
        nm_budget = self.budget - calls
        
        if nm_budget > 0:
            simplex = [best_x.copy()]
            for i in range(n):
                x_i = best_x.copy()
                step = 0.1 if best_x[i] != 0 else 1.0
                x_i[i] += step
                simplex.append(x_i)
            
            simplex_f = np.array([func(x) for x in simplex])
            calls += n
            
            while calls < self.budget:
                if time.time() > deadline:
                    break
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = simplex_f[order]
                
                centroid = np.mean(simplex[:n], axis=0)
                
                xr = centroid + (centroid - simplex[-1])
                fr = func(xr)
                calls += 1
                
                if fr < simplex_f[0]:
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    fc = func(xc)
                    calls += 1
                    
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                        calls += n
                        
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx]

        return best_x, best_f
