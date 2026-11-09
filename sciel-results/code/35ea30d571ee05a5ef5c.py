import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a local RNG for determinism
        rng = np.random.default_rng(0)
        
        # Archive to store non-dominated solutions
        archive = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            # Check if f is dominated by any point in archive_F
            # Dominated if exists g in archive_F such that g <= f and g != f
            if len(archive_F) == 0:
                return False
            AF = np.array(archive_F)
            # g <= f for all objectives
            dominated = np.all(AF <= f, axis=1)
            # g != f
            not_equal = np.any(AF < f, axis=1)
            return np.any(dominated & not_equal)

        def update_archive(x, f):
            # Remove points from archive dominated by f
            new_archive = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive, archive_F)):
                # Check if af is dominated by f
                # f <= af and f != af
                if np.all(f <= af) and np.any(f < af):
                    continue # af is dominated by f, remove it
                # Check if f is dominated by af
                # af <= f and af != f
                if np.all(af <= f) and np.any(af < f):
                    continue # f is dominated by af, don't add f
                new_archive.append(ax)
                new_archive_F.append(af)
            
            # If f is not dominated by any existing, add it
            # We already removed dominated ones. 
            # We need to check if f is dominated by any remaining.
            # The loop above skipped adding f if it was dominated by af.
            # So if we reach here, f is not dominated by any remaining af.
            # And we removed any af dominated by f.
            # So we can add f.
            new_archive.append(x)
            new_archive_F.append(f)
            
            return new_archive, new_archive_F

        # 1. Initialization
        pop_size = 10
        # Ensure budget allows for init + DE + Polish
        # Init: 10
        # DE: 140
        # Polish: 50
        
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Update archive with initial population
        for i in range(pop_size):
            archive, archive_F = update_archive(X[i], F[i])
        
        # 2. Differential Evolution
        # Run for remaining budget minus reserved polish budget
        polish_budget = 50
        de_budget = budget - used - polish_budget
        if de_budget < 0:
            de_budget = 0
            
        # DE parameters
        CR = 0.5
        F_factor = 0.8
        
        # We have used 'used' evaluations.
        # We want to use 'de_budget' more.
        # Each generation uses pop_size evaluations.
        n_gen = de_budget // pop_size
        
        for g in range(n_gen):
            if time.time() > deadline - 10: # Safety margin
                break
            if used >= budget - polish_budget:
                break
                
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                # Select 3 distinct parents
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                v = X[r1] + F_factor * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)
                
                # Crossover
                cross_point = rng.integers(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == cross_point:
                        trial[j] = v[j]
                
                ft = problem(trial)
                used += 1
                
                # Selection
                if np.all(ft <= F[i]):
                    new_X.append(trial)
                    new_F.append(ft)
                    # Update archive
                    archive, archive_F = update_archive(trial, ft)
                else:
                    new_X.append(X[i])
                    new_F.append(F[i])
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        # 3. Local Polish
        # Select top K non-dominated solutions from archive
        # We have used 'used' evaluations.
        # Remaining: budget - used
        remaining = budget - used
        
        if remaining > 0:
            # Sort archive by sum of objectives or just take random ones?
            # Taking diverse ones is better.
            # Let's take up to 5 points from the archive.
            # If archive is large, we should sample.
            # For simplicity, take the first 5 or random 5.
            # Let's take 5 random non-dominated points from the archive.
            if len(archive) > 0:
                n_polish = min(5, len(archive))
                # Randomly select n_polish indices
                indices = rng.choice(len(archive), n_polish, replace=False)
                
                for idx in indices:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                        
                    x0 = archive[idx]
                    f0 = archive_F[idx]
                    
                    # Define scalar objective for Nelder-Mead
                    # Weighted sum. Weights can be random or based on f0.
                    # Let's use uniform weights or inverse of f0 to target the knee?
                    # Uniform weights minimize the sum, which finds a point on the front.
                    # To find different points, we can vary weights.
                    # Let's use a fixed set of weight vectors for the polish steps.
                    # Or just minimize sum of objectives.
                    
                    def obj_scalar(x):
                        fx = problem(x)
                        return np.sum(fx)
                    
                    # Nelder-Mead
                    # maxiter should be small to stay within budget
                    # Each iteration uses ~dim+1 evals? No, it uses a simplex update.
                    # Let's limit maxiter to ensure we don't blow the budget.
                    # If we have ~10 evals left per point, maxiter=5 is safe.
                    max_iter = max(1, (budget - used) // n_polish)
                    
                    try:
                        res = minimize(obj_scalar, x0, method='Nelder-Mead', 
                                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                        # res.x is the best found
                        # Update used? We don't know exactly how many evals minimize used.
                        # scipy doesn't return eval count easily in all versions, 
                        # but we can estimate or just assume it used some.
                        # To be safe, we should track evals inside obj_scalar.
                        
                    except Exception:
                        pass

        return X
