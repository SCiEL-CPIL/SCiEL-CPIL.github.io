"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class CMAES:
    def __init__(self, x0, sigma0, dim, budget, rng, func, deadline):
        self.dim = dim
        self.x0 = x0
        self.sigma = sigma0
        self.budget = budget
        self.rng = rng
        self.func = func
        self.deadline = deadline
        
        # CMA-ES parameters
        self.mu = dim // 2 + 1  # number of parents
        self.lam = 2 * self.mu  # population size
        self.c1 = (2 / ((dim + 1) ** 2 + 10))
        self.cmua = (
            (4 + dim / self.mu) / ((dim + 2) ** 2 + 12)
        )
        self.c_sigma = (self.mu + 0.5 + self.cmua) / (dim + self.mu + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu - 1) / (dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + dim / self.mu) / (dim + 4 + 4 * self.mu / dim)
        self.d_c = 1 + 2 * max(0, np.sqrt(self.mu / (dim + 1) - 1)) + self.c_c
        self.cs = self.c_sigma
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu - 1) / (dim + 1)) - 1) + self.cs
        
        # State
        self.x_mean = x0.copy()
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))
        
        self.calls_used = 0
        self.best_x = x0.copy()
        self.best_f = self.func(self.best_x)
        self.calls_used += 1
        self.stall_count = 0
        self.last_improvement = self.best_f

    def step(self):
        if self.calls_used >= self.budget or time.time() > self.deadline:
            return False
            
        # Sample population
        B = self.rng.standard_normal((self.lam, self.dim))
        # Whitening transformation: C^(1/2) * B
        # Use eigendecomposition for stability
        D, V = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 1e-12))
        Y = (B * D) @ V.T
        Y = Y * self.sigma
        
        x_pop = self.x_mean + Y
        # Clip to box
        x_pop = np.clip(x_pop, -5, 5)
        
        # Evaluate
        f_pop = np.array([self.func(x) for x in x_pop])
        self.calls_used += self.lam
        
        # Update best
        min_idx = np.argmin(f_pop)
        if f_pop[min_idx] < self.best_f:
            self.best_f = f_pop[min_idx]
            self.best_x = x_pop[min_idx].copy()
            self.stall_count = 0
            self.last_improvement = self.best_f
        else:
            self.stall_count += 1
        
        # Sort
        o = np.argsort(f_pop)
        x_sorted = x_pop[o]
        f_sorted = f_pop[o]
        
        # Weights
        w = np.log(self.mu) - np.arange(self.mu)
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Update mean
        self.x_mean = np.sum(w[:, None] * x_sorted[:self.mu], axis=0)
        
        # Update step size control variables
        y_old = (self.x_mean - self.x_mean - Y[o[0]]) / self.sigma  # This is wrong, need old mean
        # Correct update logic:
        # y_k = (x_mean_new - x_mean_old) / sigma
        y = (self.x_mean - self.x_mean_old) / self.sigma if hasattr(self, 'x_mean_old') else np.zeros(self.dim)
        
        # Actually, standard CMA-ES updates:
        # 1. Save old mean
        # 2. Compute y
        # 3. Update ps, pc, C, sigma
        
        # Let's re-implement step carefully to avoid state errors
        
        return True

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Multi-start strategy: Run several independent CMA-ES instances
        # or a single one with restarts. 
        # Given dim=5, budget=300, a single CMA-ES with lambda=12 uses ~25 generations.
        # Multi-start with 5 instances of 60 calls each? Or 1 instance with aggressive restart?
        
        # Let's try a robust (1+1)-ES first as it's simpler and less prone to implementation bugs, 
        # but enhanced with multi-starts and better sigma adaptation.
        # Actually, a simple population-based ES is safer.
        
        def run_es(x_start, sigma_start, budget_es):
            x = x_start.copy()
            fx = func(x)
            best_x_local = x.copy()
            best_f_local = fx
            sigma = sigma_start
            calls = 1
            stall = 0
            threshold = 20
            
            while calls < budget_es and time.time() < deadline:
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stall = 0
                    if fx < best_f_local:
                        best_x_local = x.copy()
                        best_f_local = fx
                else:
                    sigma *= 0.9
                    stall += 1
                
                if stall > threshold:
                    break
                    
            return best_x_local, best_f_local, calls

        # Strategy: 
        # 1. One main run with large budget
        # 2. If it stalls, use remaining budget for random restarts around best
        
        x0 = rng.uniform(-5, 5, self.dim)
        x_best, f_best, used = run_es(x0, 2.0, self.budget)
        
        # If we have budget left, do random restarts
        remaining = self.budget - used
        if remaining > 10:
            for _ in range(5):
                if remaining <= 10 or time.time() > deadline:
                    break
                x_rand = np.clip(x_best + rng.standard_normal(self.dim) * 1.0, -5, 5)
                x_r, f_r, used_r = run_es(x_rand, 1.0, remaining)
                remaining -= used_r
                if f_r < f_best:
                    x_best, f_best = x_r, f_r

        return x_best, f_best
# EVOLVE-BLOCK-END
