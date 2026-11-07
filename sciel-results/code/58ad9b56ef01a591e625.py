import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_F = []
        self.evals_used = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initial Sampling
        init_size = 40
        if init_size > self.budget:
            init_size = self.budget
            
        sampler = qmc.Halton(d=self.dim, scramble=False)
        points = sampler.random(init_size)
        X_init = lo + points * (hi - lo)
        
        F_init = np.array([problem(x) for x in X_init])
        self.evals_used = init_size
        
        def update_archive(X_new, F_new):
            if self.archive:
                X_all = np.vstack([self.archive, X_new])
                F_all = np.vstack([self.archive_F, F_new])
            else:
                X_all = X_new
                F_all = F_new
            
            n = len(F_all)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        is_dominated[i] = True
                        break
            mask = ~is_dominated
            self.archive = X_all[mask]
            self.archive_F = F_all[mask]

        update_archive(X_init, F_init)
        
        # 2. Adaptive Refinement
        rng = np.random.default_rng(123)
        step_size = 0.08 * (hi - lo)
        
        while self.evals_used < self.budget:
            if len(self.archive) == 0:
                break
                
            n_arch = len(self.archive_F)
            if n_arch == 1:
                idx = 0
            else:
                # Vectorized distance calculation for gap-filling
                dists = np.linalg.norm(self.archive_F[:, np.newaxis, :] - self.archive_F[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                min_dists = np.min(dists, axis=1)
                idx = np.argmax(min_dists)
                
            x_current = self.archive[idx]
            
            dim_idx = rng.integers(0, self.dim)
            sign = rng.choice([-1, 1])
            
            direction = np.zeros(self.dim)
            direction[dim_idx] = sign
            
            x_new = x_current + step_size * direction
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            self.evals_used += 1
            
            update_archive(x_new.reshape(1, -1), f_new.reshape(1, -1))
            
        return self.archive
