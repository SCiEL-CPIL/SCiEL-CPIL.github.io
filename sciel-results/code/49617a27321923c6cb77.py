"""Hybrid DE-CMA-ES optimizer for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Differential Evolution (Global Search)
        # Use ~30% of budget for global exploration
        de_budget = int(budget * 0.3)
        if de_budget < 20: de_budget = 20
        if de_budget > budget - 20: de_budget = budget - 20
        
        best_x, best_f = self._run_de(func, de_budget, dim)
        
        # Check time
        if time.time() > deadline - 10.0:
            return best_x, best_f

        # Phase 2: CMA-ES (Local Refinement)
        remaining_budget = budget - de_budget
        # CMA-ES needs at least a few generations
        if remaining_budget < 10:
            return best_x, best_f
            
        best_x, best_f = self._run_cma_es(func, remaining_budget, dim, best_x, best_f)
        
        return best_x, best_f

    def _run_de(self, func, budget, dim):
        """Simple Differential Evolution for global search."""
        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        pop_size = max(10, int(5 * dim))  # Population size
        
        # Initialize population uniformly in [-5, 5]
        # Use a fixed seed for reproducibility? No, random is fine, but let's use a random seed based on time for variety
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        fits = np.array([func(x) for x in pop])
        calls = pop_size
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        while calls + pop_size <= budget:
            if time.time() > self.deadline_check():
                break
                
            new_pop = np.zeros((pop_size, dim))
            new_fits = np.zeros(pop_size)
            
            for i in range(pop_size):
                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
        return best_x, best_f

    def _run_cma_es(self, func, budget, dim, x_start, f_start):
        """CMA-ES for local refinement."""
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        x = x_start.copy()
        # Start with a smaller sigma since we are in a local refinement phase
        # Estimate sigma from the distance of recent DE points? Or just a small value.
        sigma = 0.5 
        
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_f = f_start
        calls = 0
        
        # We don't count the initial x_start as a call in this phase if it was already evaluated in DE
        # But to be safe, we treat this as a fresh start for CMA-ES accounting.
        
        while calls + lam <= budget:
            if time.time() > self.deadline_check():
                break
                
            # Sample
            ys = np.zeros((lam, dim))
            # Compute Cholesky once per generation
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            for i in range(lam):
                z = np.random.standard_normal(dim)
                ys[i] = x + sigma * (L @ z)
                ys[i] = np.clip(ys[i], -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls += lam
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = ys[idx_min].copy()
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Mean update
            x_old = x.copy()
            x_new = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Step size adaptation
            dx = x_new - x_old
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ dx) / sigma
                # h_sigma calculation
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.trace(np.eye(dim)) else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.trace(np.eye(dim))) - 1))
            else:
                h_sigma = 1
            
            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * dx / sigma
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i] - x_old, ys[i] - x_old)
            
            C = (1 + c_1 * h_sigma * (dim + 1) - c_1 - c_mu) * C + c_1 * (rank_one - (1 - h_sigma) * (1 - c_c) * C) + c_mu / sigma**2 * rank_mu
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            try:
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-16, None)
                C_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            x = x_new
            
            if sigma < 1e-12:
                break

        return best_x, best_f

    def deadline_check(self):
        return time.time() > 290.0
# EVOLVE-BLOCK-END
