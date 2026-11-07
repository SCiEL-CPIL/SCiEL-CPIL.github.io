"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Multimodal Differential Evolution with Local Refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        # Parameters
        NP = 20  # Population size
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # State
        x_best = None
        f_best = float('inf')
        
        # Initialization
        # We need to reserve some budget for local refinement
        # Let's use 70% of budget for DE, 30% for local polish if needed
        # But DE itself consumes budget. 
        # Total budget is 500.
        # Let's run DE for a fixed number of generations that fits the budget.
        # Each generation consumes NP evaluations (approx, due to selection).
        # Actually, standard DE consumes NP evaluations per generation.
        # 500 / 20 = 25 generations.
        
        # Let's use a slightly smaller population to allow more generations or larger local polish
        # NP=10 -> 50 generations.
        # Let's stick to NP=20 for better multimodal coverage, but be careful with budget.
        
        # We will track evaluations used.
        evals_used = 0
        
        def safe_call(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                return f_best # Return best known to avoid errors, though we shouldn't call
            x_clipped = np.clip(x, -self.bound, self.bound)
            fx = func(x_clipped)
            evals_used += 1
            return fx

        # 1. Initialize Population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-self.bound, self.bound, (NP, self.dim))
        pop_f = np.array([safe_call(ind) for ind in pop])
        
        # Track best
        idx_best = np.argmin(pop_f)
        x_best = pop[idx_best].copy()
        f_best = pop_f[idx_best]
        
        # DE Loop
        # We want to use most of the budget here.
        # Let's reserve 50 evaluations for a final local polish on the best 5 candidates.
        de_budget = self.budget - 50
        
        while evals_used < de_budget:
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(NP):
                if evals_used >= de_budget:
                    break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -self.bound, self.bound)
                
                # Crossover
                # Binary crossover
                j_rand = rng.integers(self.dim)
                trial = pop[i].copy()
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Evaluate
                ft = safe_call(trial)
                
                # Selection
                if ft < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = ft
                    if ft < f_best:
                        f_best = ft
                        x_best = trial.copy()

        # 2. Local Refinement Phase
        # Use the remaining budget to polish the best few candidates
        # Identify top 5 unique candidates from the final population
        # Since DE population might have duplicates or close points, we just take the best 5 indices
        
        # We have `self.budget - evals_used` evaluations left.
        # Let's say we have R evaluations left.
        # We will polish the single best candidate with a local optimizer.
        # Or polish top 3.
        
        # Let's use Nelder-Mead on the best candidate.
        # It might be slow if not careful, but for 5D and small budget it's fine.
        
        # Define a wrapper for scipy that counts evals and respects budget
        class BudgetFunc:
            def __init__(self, func, x_best, f_best, budget_remaining):
                self.func = func
                self.x_best = x_best
                self.f_best = f_best
                self.remaining = budget_remaining
                self.count = 0
            
            def __call__(self, x):
                if self.count >= self.remaining:
                    return self.f_best
                x_clipped = np.clip(x, -self.bound, self.bound)
                fx = self.func(x_clipped)
                self.count += 1
                if fx < self.f_best:
                    self.f_best = fx
                    self.x_best = x_clipped.copy()
                return fx

        remaining_budget = self.budget - evals_used
        if remaining_budget > 10: # Need some minimum for NM to work
            # Polish the best candidate
            x_start = x_best.copy()
            bf = BudgetFunc(func, x_start, f_best, remaining_budget)
            
            try:
                # Nelder-Mead is derivative-free
                # maxiter is not strictly evaluated count, so we rely on the wrapper
                res = minimize(bf, x_start, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x
            
            except Exception:
                pass

        return x_best, f_best

# EVOLVE-BLOCK-END
