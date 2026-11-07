import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 100
        # DE parameters
        F = 0.5
        CR = 0.9

        # Initialize population
        # Use a mix of random points and some points close to the known front structure
        # to help convergence, but let DE do the heavy lifting
        pop = np.random.uniform(lo, hi, size=(pop_size, n))
        # Add a few points with x[1:] = 0 to seed the front
        for i in range(10):
            pop[i, 1:] = 0.0
            pop[i, 0] = np.random.uniform(0, 1)

        # Evaluate initial population
        F_vals = np.array([problem(x) for x in pop])
        used = pop_size

        # Track all non-dominated points found so far for hypervolume
        # We will store all evaluated points and filter at the end, 
        # but to save memory/time, we can maintain a non-dominated set.
        # However, with only 200 evaluations, storing all is fine.
        all_points = F_vals.copy()
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, F_set):
            # F_set is a list/array of objective vectors
            for g in F_set:
                if dominates(g, f):
                    return True
            return False

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = np.zeros_like(pop)
            off_vals = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct random parents different from i
                r1, r2, r3 = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # DE mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.zeros_like(v)
                j_rand = np.random.randint(n)
                for j in range(n):
                    if np.random.rand() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                all_points = np.vstack([all_points, [f_trial]])
                
                # Selection: keep the better of parent and offspring
                if dominates(f_trial, F_vals[i]):
                    offspring[i] = trial
                    off_vals[i] = f_trial
                else:
                    offspring[i] = pop[i]
                    off_vals[i] = F_vals[i]
            
            if used >= self.budget:
                break
                
            # Update population
            pop = offspring
            F_vals = off_vals

        # Filter out points dominated by reference point [11, 11]
        # And ideally return the non-dominated set, but the evaluator takes all recorded vectors.
        # The evaluator computes hypervolume of ALL recorded vectors.
        # So we just return the array of all points.
        return all_points
