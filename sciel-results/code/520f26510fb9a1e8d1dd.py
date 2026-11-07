"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing


class Optimizer:
    """Population-based restart strategy with dual annealing for robustness."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # We have 500 evaluations total.
        # We will use a restart strategy.
        # Let's divide the budget into blocks.
        # Each block runs a local search (dual_annealing or similar) with a limited budget.
        
        # However, dual_annealing has its own internal budget. 
        # We need to be careful how we map our 500 evals to multiple runs.
        
        # Strategy:
        # 1. Initial phase: A few random starts with a local optimizer to find good basins.
        # 2. Refinement: Run a high-budget optimizer from the best point found.
        
        # Since Rastrigin is multimodal, global search is needed.
        # dual_annealing is a good global optimizer.
        
        # Let's use a simple approach:
        # Split 500 evals into N restarts.
        # For each restart, run a local search with budget B.
        # Keep the best point.
        
        # But dual_annealing might not strictly respect a small per-call budget in a way that allows easy restarts without overhead.
        # Let's try a CMA-ES like approach or just a robust multi-start local search.
        
        # Actually, for dim=5 and 500 evals, a single run of a good global optimizer like CMA-ES or DE might be sufficient if it's efficient.
        # But CMA-ES population size is usually lambda = 4 + floor(3 ln n). For n=5, lambda = 4 + 3*1.6 = 8.8 -> 12.
        # 500 / 12 ~ 40 generations. Might be too few for global convergence on Rastrigin.
        
        # Let's use a multi-start strategy with scipy.optimize.differential_evolution or dual_annealing.
        # dual_annealing is often very fast and effective for small dims.
        
        # We can run dual_annealing with maxiter set such that it uses up the budget.
        # But we don't know the exact cost per iteration.
        
        # Alternative: Implement a simple (1+lambda)-ES or DE with restarts.
        
        # Let's try a Population Based Incremental Learning (PBIL) or simple DE.
        
        # Simple Differential Evolution (DE/rand/1/bin)
        # Pop size NP = 10
        # F = 0.8, CR = 0.9
        # 500 evals / 10 = 50 generations.
        # This is a reasonable number for DE in 5D.
        
        def de_search(func, budget, dim, rng, x0=None):
            NP = 10
            F = 0.8
            CR = 0.9
            lb = -5.0
            ub = 5.0
            
            # Initialize population
            if x0 is not None:
                pop = np.random.uniform(lb, ub, (NP, dim), generator=rng)
                pop[0] = x0
            else:
                pop = np.random.uniform(lb, ub, (NP, dim), generator=rng)
                
            fvals = np.array([func(x) for x in pop])
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]
            
            evals_used = NP
            max_evals = budget
            
            while evals_used < max_evals:
                # Check time
                if time.time() > deadline:
                    break
                
                # Select base vector
                i = rng.integers(NP)
                # Choose 3 distinct random indices
                a, b, c = np.random.choice(NP, 3, replace=False, generator=rng)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip
                trial = np.clip(trial, lb, ub)
                
                # Evaluate
                trial_f = func(trial)
                evals_used += 1
                
                # Selection
                if trial_f <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        
            return best_x, best_f

        rng = np.random.default_rng(42)
        
        # We have 500 evals.
        # Let's do 5 restarts of 100 evals each? Or one big run?
        # DE with NP=10 uses 10 evals for init. Then 1 eval per generation per individual? No, 1 trial per individual per generation.
        # So per generation, we use 10 evals.
        # 100 evals -> ~10 generations. Too few.
        
        # Let's use a smaller population for more generations?
        # NP=5. 500/5 = 100 generations. Better.
        
        # Or use a (1+1) ES with restarts?
        
        # Let's stick to DE with NP=10 but run it for the full budget if it's the only run.
        # 500 evals -> 50 generations.
        
        # To improve global search, let's do multiple restarts.
        # Restart 1: 100 evals
        # Restart 2: 100 evals
        # ...
        # Restart 5: 100 evals
        
        # But restarting DE from scratch loses memory.
        # Better: Run one DE for 500 evals. It's a population method, so it explores.
        
        # Let's try one DE run with NP=20?
        # 500/20 = 25 generations.
        
        # Let's try a hybrid:
        # Phase 1: 200 evals with DE (NP=10, 20 gens)
        # Phase 2: 300 evals with a local search (CMA-ES or BFGS) from the best point found in Phase 1.
        
        # CMA-ES is very effective for local refinement.
        # We can implement a simple CMA-ES or use scipy.optimize.
        
        # Let's use scipy.optimize.minimize with method='Nelder-Mead' or 'Powell' for local refinement?
        # They don't have a strict budget control easily.
        
        # Let's implement a simple (1+1)-ES for the refinement phase.
        
        def es_refine(func, x_start, budget, dim, rng, deadline):
            x = x_start.copy()
            fx = func(x)
            sigma = 1.0
            evals_used = 1
            
            while evals_used < budget:
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                    
            return x, fx

        # Phase 1: DE
        # Budget 1: 200
        best_x, best_f = de_search(func, 200, self.dim, rng)
        
        # Phase 2: ES Refinement
        # Budget 2: 300
        final_x, final_f = es_refine(func, best_x, 300, self.dim, rng, deadline)
        
        return final_x, final_f

# EVOLVE-BLOCK-END
