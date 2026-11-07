import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Initialize
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Multi-resolution Scan (Halton Sequence)
        # The Step Ellipsoidal function has a flat plateau. 
        # A systematic scan ensures we cross the discontinuity early.
        # We use a Halton sequence to cover the space uniformly.
        
        # Let's use 30% of the budget for this scan.
        scan_budget = int(budget * 0.3)
        
        # We want to generate points in [0, 1]^d and map them to [-5, 5]^d
        # Halton sequence requires prime bases.
        # For d=5, primes are [2, 3, 5, 7, 11]
        
        # We will generate points in batches to check time/budget
        # Start with a coarse resolution, then refine?
        # Actually, just streaming a Halton sequence is good for coverage.
        
        # To ensure we don't miss the "step" boundary, we should perhaps
        # stratify or use a specific order. Halton is low-discrepancy.
        
        # Let's generate the first 'scan_budget' points of a Halton sequence.
        # However, generating all at once might be memory heavy if budget is large,
        # but 150 points in 5D is trivial.
        
        # Note: qmc.Halton requires d dimensions.
        # We need to be careful with the first point (0,0,0... -> -5,-5,-5...)
        # and subsequent points.
        
        # We'll use a simple loop to generate and evaluate one by one to strictly respect budget/time.
        # qmc.Halton can generate a stream, but it's easier to just use the next() method or generate chunks.
        # Let's generate chunks of 10.
        
        halton = qmc.Halton(d=n, scramble=True, seed=42)
        # We need to skip the first point if it's the origin? 
        # Halton usually starts at 0. 0 maps to -5. 
        # It's fine.
        
        points_generated = 0
        while calls_made < scan_budget and (time.time() - start_time) < deadline:
            # Generate a small batch
            batch_size = min(10, scan_budget - calls_made)
            if batch_size <= 0:
                break
            
            # Generate points in [0,1]^d
            points_unit = halton.random(batch_size)
            
            # Map to [-5, 5]^d
            # x = 10 * u - 5
            points_mapped = points_unit * 10.0 - 5.0
            
            for p in points_mapped:
                if calls_made >= scan_budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                evaluate(p)
            
            # Check if we ran out of time/budget
            if (time.time() - start_time) > deadline:
                break

        # Phase 2: Differential Evolution (DE) for Global Search
        # Use the best point from the scan as an anchor.
        # DE is good for escaping local flat regions if not converged.
        
        de_budget_limit = int(budget * 0.7)
        
        if calls_made < de_budget_limit:
            pop_size = 10
            if pop_size < 4:
                pop_size = 4
            
            # Initialize population
            # Mix of random points and points around the current best
            # 1 point at best_x
            # Rest random in [-5, 5] or around best_x?
            # Since we did a scan, best_x is likely decent.
            # Let's initialize 1 at best_x, and the rest randomly, 
            # but maybe bias towards the region around best_x to exploit the "step" found?
            # No, DE needs diversity. Random is fine, but let's ensure best_x is in the pop.
            
            population = rng.uniform(-5, 5, (pop_size, n))
            population[0] = best_x
            
            fitness = np.zeros(pop_size)
            
            # Evaluate population
            # We already evaluated best_x? Yes, in Phase 1.
            # But we need to track fitness array for DE selection.
            # We can re-evaluate best_x to be safe, or just store it.
            # Re-evaluating wastes 1 call. Let's just assume best_x is in pop[0] and set fitness[0] = best_f.
            # But we need to evaluate the other pop_size - 1 points.
            
            fitness[0] = best_f
            for i in range(1, pop_size):
                if calls_made >= de_budget_limit:
                    break
                f_val = func(population[i])
                fitness[i] = f_val
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = population[i].copy()
            
            # DE Parameters
            F = 0.8
            CR = 0.9
            
            while calls_made < de_budget_limit:
                if (time.time() - start_time) > deadline:
                    break
                
                for i in range(pop_size):
                    if calls_made >= de_budget_limit:
                        break
                    
                    # Select three distinct random individuals
                    # Ensure they are not i
                    candidates = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    
                    # Mutation
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Crossover
                    trial = np.zeros(n)
                    j_rand = rng.integers(n)
                    for j in range(n):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]
                            
                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate trial
                    f_trial = func(trial)
                    calls_made += 1
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
                    # Selection
                    if f_trial <= fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                
                if (time.time() - start_time) > deadline:
                    break

        # Phase 3: Local Refinement (Coordinate Descent)
        # Use the remaining budget.
        # The Step Ellipsoidal function is non-smooth.
        # Coordinate descent with shrinking steps is effective for finding the "corner" of the ellipsoid.
        
        if calls_made < budget:
            step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
            
            current_x = best_x.copy()
            current_f = best_f
            
            for step in step_sizes:
                if calls_made >= budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                    
                improved = True
                # Limit iterations per step size to avoid infinite loops if no improvement
                max_iter = 20
                iter_count = 0
                
                while improved and calls_made < budget and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for d_idx in range(n):
                        if calls_made >= budget:
                            break
                        if (time.time() - start_time) > deadline:
                            break
                            
                        # Try positive direction
                        x_pos = current_x.copy()
                        x_pos[d_idx] += step
                        x_pos = np.clip(x_pos, -5, 5)
                        f_pos = func(x_pos)
                        calls_made += 1
                        
                        if f_pos < current_f:
                            current_x = x_pos
                            current_f = f_pos
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue # Restart coordinate descent loop? 
                            # Actually, 'continue' skips the negative check for this dimension.
                            # This is a "first improvement" strategy.
                            
                        # Try negative direction
                        x_neg = current_x.copy()
                        x_neg[d_idx] -= step
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        calls_made += 1
                        
                        if f_neg < current_f:
                            current_x = x_neg
                            current_f = f_neg
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue

        return best_x, best_f
