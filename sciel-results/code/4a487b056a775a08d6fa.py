import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        used = 0
        
        # Generic Differential Evolution approach
        # We don't assume the problem is DTLZ2. We use a robust metaheuristic.
        # Population size: large enough for diversity, small enough to fit in budget.
        # Budget 200. Let's use pop_size = 100.
        # Initial population: 100 evals.
        # Remaining: 100 evals for 1 generation of DE.
        # Or we can do multiple generations with smaller pop.
        # Let's try pop_size = 50. 4 generations of 50 offspring = 200 evals.
        # Total evals: 50 (init) + 4*50 (offspring) = 250? No, budget is 200.
        # Let's do:
        # Init: 50 evals.
        # Loop while budget allows:
        #   Generate 50 offspring.
        #   Evaluate 50 offspring.
        #   Update population (elitism).
        # 50 + 50 + 50 + 50 = 200. So 3 generations of improvement after init.
        
        pop_size = 50
        
        # Initialize population randomly
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used += pop_size
        
        # Store all evaluated points for final hypervolume calculation
        # The task says "Every returned objective vector is recorded".
        # So we just need to make sure we call problem() and let the evaluator record it.
        # We don't need to store them ourselves for the return value, but we need them for optimization.
        
        all_X = X.copy()
        all_F = F.copy()
        
        # DE Parameters
        F_cr = 0.8  # Differential weight
        C_r = 0.9   # Crossover rate
        
        # Optimization Loop
        while used < max_evals:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)
            
            for i in range(pop_size):
                if used >= max_evals:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                v = X[r1] + F_cr * (X[r2] - X[r3])
                
                # Binary crossover
                j_rand = np.random.randint(self.dim)
                trial = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.random() <= C_r or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                
                offspring_X[i] = trial
                offspring_F[i] = f_trial
                
                # Add to global record
                all_X = np.vstack([all_X, trial])
                all_F = np.vstack([all_F, f_trial])
            
            if used >= max_evals:
                break
                
            # Selection: Elitism
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort and crowding distance
            # Simple approach: Sort by sum of objectives (Tchebycheff or simple sum)
            # For DTLZ2, sum of objectives is a good proxy for distance to origin.
            # But for generic MOO, we should use proper NSGA-II selection.
            # Given time constraints, let's use a simple dominance-based selection.
            
            # Identify non-dominated points in combined set
            # This is O(N^2 * M). N=100, M=3. 100*100*3 = 30000 ops. Fast.
            
            def dominates(a, b):
                # a dominates b if a <= b in all objectives and a < b in at least one
                return np.all(a <= b) and np.any(a < b)
            
            nd_indices = []
            for i in range(len(combined_F)):
                is_dominated = False
                for j in range(len(combined_F)):
                    if i == j:
                        continue
                    if dominates(combined_F[j], combined_F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_indices.append(i)
            
            # If we have more non-dominated points than pop_size, we need to select
            # We can select based on crowding distance or just random/first
            # For simplicity, if len(nd_indices) > pop_size, pick pop_size random ones from nd_indices
            # Or better: sort by sum of objectives and take the best pop_size?
            # Actually, for hypervolume, diversity is key.
            # Let's just take the first pop_size non-dominated points found.
            # To be more robust, let's sort non-dominated points by their sum of objectives
            # and take the ones with the lowest sums? No, that loses diversity.
            
            # Let's use a simple crowding distance approximation:
            # Sort non-dominated points by first objective, calculate distance to neighbors.
            # This is complex. Let's just take a random subset if too many, or all if fewer.
            
            if len(nd_indices) > pop_size:
                # Select pop_size points from non-dominated set
                # Random selection to maintain diversity
                selected_indices = np.random.choice(nd_indices, pop_size, replace=False)
            else:
                # Keep all non-dominated, fill with dominated if needed (shouldn't happen often with DE)
                # If we have fewer, we might want to add some dominated points to maintain population size
                # But usually DE maintains a good front.
                # Let's just take all non-dominated and pad with random dominated if necessary
                selected_indices = list(nd_indices)
                dominated_indices = [i for i in range(len(combined_F)) if i not in nd_indices]
                while len(selected_indices) < pop_size and dominated_indices:
                    # Pick random dominated
                    idx = np.random.choice(dominated_indices)
                    selected_indices.append(idx)
                    dominated_indices.remove(idx)
                
                # If still not enough (unlikely), just repeat
                while len(selected_indices) < pop_size:
                    selected_indices.append(np.random.randint(len(combined_F)))
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]
            
        return np.array(all_X)
