"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for RE21 using Differential Evolution with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        archive_X = []
        archive_F = []
        used = 0

        def evaluate_and_add(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            
            if len(archive_F) == 0:
                archive_X.append(x)
                archive_F.append(f)
                return f

            F_arr = np.array(archive_F)
            
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if np.any(mask_le & mask_lt):
                return None  # Dominated, discard

            mask_le2 = np.all(f <= F_arr, axis=1)
            mask_lt2 = np.any(f < F_arr, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]
            
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
            
            archive_X.append(x)
            archive_F.append(f)
            return f

        # Phase 1: Initialization
        n_init = 20
        X_init = lo + span * np.random.random((n_init, dim))
        
        for x in X_init:
            evaluate_and_add(x)

        # Phase 2: Differential Evolution
        # Reserve 80 evaluations for local polish
        reserve = 80
        pop_size = 10  # Small population for 4D problem
        
        # DE parameters
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Main DE loop
        while used < budget - reserve:
            if time.time() > deadline:
                break
            
            # Create population from archive + random fill
            pop_X = []
            pop_F = []
            
            # Add archive points
            for i in range(len(archive_X)):
                if len(pop_X) < pop_size:
                    pop_X.append(np.array(archive_X[i]))
                    pop_F.append(np.array(archive_F[i]))
            
            # Fill remaining with random points
            while len(pop_X) < pop_size:
                x_rand = lo + span * np.random.random(dim)
                f_rand = evaluate_and_add(x_rand)
                if f_rand is not None:
                    pop_X.append(x_rand)
                    pop_F.append(f_rand)
                else:
                    # If dominated, still add to population for DE to work with
                    pop_X.append(x_rand)
                    pop_F.append(np.array(problem(x_rand), dtype=np.float64) if used < budget else np.array([3000.0, 0.06]))
                    if used < budget:
                        used += 1
            
            if len(pop_X) < 3:
                break
            
            pop_X = np.array(pop_X)
            pop_F = np.array(pop_F)
            
            # DE iteration: generate trial vectors
            n_trials = 0
            max_trials = pop_size
            
            for i in range(pop_size):
                if used >= budget - reserve:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop_X[a] + F * (pop_X[b] - pop_X[c])
                
                # Crossover: binomial crossover
                trial = pop_X[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, lo, hi)
                
                # Evaluate trial
                f_trial = evaluate_and_add(trial)
                n_trials += 1
                
                # If trial was added to archive (non-dominated), it's good
                # If not added (dominated), we could still use it for local search later
        
        # Phase 3: Local Polish
        # Use remaining budget for local optimization of archive points
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            
            # Select points to polish: spread along the front
            order = np.argsort(F_arr[:, 0])
            n = len(order)
            
            # Select up to 5 points evenly spaced
            if n <= 5:
                indices_to_optimize = list(range(n))
            else:
                indices_to_optimize = [
                    order[0],
                    order[n//4],
                    order[n//2],
                    order[3*n//4],
                    order[-1]
                ]
            
            # Distribute remaining budget among selected points
            remaining = budget - used
            if remaining > 0 and len(indices_to_optimize) > 0:
                evals_per_point = max(5, remaining // len(indices_to_optimize))
                
                for idx in indices_to_optimize:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x0 = np.array(archive_X[idx])
                    f0 = np.array(archive_F[idx])
                    
                    # Use weighted sum with weights based on point position
                    # Normalize weights to sum to 1
                    w1 = 1.0 / max(f0[0], 1e-6)
                    w2 = 1.0 / max(f0[1], 1e-6)
                    w_sum = w1 + w2
                    w = np.array([w1/w_sum, w2/w_sum])
                    
                    def obj(x):
                        x_clipped = np.clip(x, lo, hi)
                        f = np.array(problem(x_clipped), dtype=np.float64)
                        return np.dot(w, f)
                    
                    # Run Nelder-Mead with tight tolerance
                    try:
                        res = minimize(obj, x0, method='Nelder-Mead', 
                                       options={'maxiter': 50, 'maxfev': evals_per_point, 
                                                'xatol': 1e-6, 'fatol': 1e-6,
                                                'adaptive': True})
                        
                        # Evaluate the result
                        if used < budget:
                            x_best = np.clip(res.x, lo, hi)
                            f_best = np.array(problem(x_best), dtype=np.float64)
                            used += 1
                            evaluate_and_add(x_best)
                            
                    except Exception:
                        pass

        return archive_X
# EVOLVE-BLOCK-END
