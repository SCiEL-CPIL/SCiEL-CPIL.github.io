"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 250  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Heuristic: Positioning variables (indices 2 to end) should be 0 (or closest to 0)
        x_base = np.zeros(self.dim)
        for i in range(2, self.dim):
            if lo[i] <= 0 <= hi[i]:
                x_base[i] = 0.0
            elif 0 < lo[i]:
                x_base[i] = lo[i]
            else:
                x_base[i] = hi[i]
        
        # Generate weight vectors for scalarization
        # Include extreme points (axes) and some random points on the simplex
        rng = np.random.default_rng(42)
        weights = []
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        
        # Add random weights
        for _ in range(10):
            w = rng.random(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
            
        weights = np.array(weights)
        
        archive_f = []
        archive_x = []
        evals_used = 0
        
        def get_f(x):
            nonlocal evals_used
            if evals_used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            evals_used += 1
            return f
        
        # 1. Scalarization Phase
        for w in weights:
            if evals_used >= self.budget or time.time() > deadline:
                break
            
            rem_budget = self.budget - evals_used
            if rem_budget < 5:
                break
                
            # Determine max iterations for Nelder-Mead
            # Each iteration uses ~ dim+1 evaluations
            max_iter = max(1, rem_budget // (self.dim + 1))
            
            # Initial guess: vary deciding variables (0, 1)
            x0 = x_base.copy()
            x0[0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            x0[1] = lo[1] + (hi[1] - lo[1]) * rng.random()
            
            def scalar_obj(x):
                nonlocal evals_used
                if evals_used >= self.budget or time.time() > deadline:
                    return 0.0
                f = problem(x)
                evals_used += 1
                return np.dot(w, f)
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                # Evaluate the final point if it wasn't the last evaluation
                # Nelder-Mead returns the best point found
                if evals_used < self.budget and time.time() < deadline:
                    f_final = problem(res.x)
                    evals_used += 1
                    archive_f.append(f_final)
                    archive_x.append(res.x)
            except Exception:
                pass

        # 2. Local Refinement Phase
        # Keep non-dominated points from the archive
        # And use remaining budget to refine them
        
        # First, filter non-dominated
        non_dom_indices = []
        for i in range(len(archive_f)):
            is_dom = False
            for j in range(len(archive_f)):
                if i == j: continue
                if np.all(archive_f[j] <= archive_f[i]) and np.any(archive_f[j] < archive_f[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom_indices.append(i)
        
        if not non_dom_indices:
            # Fallback: return best point by sum
            if archive_f:
                best_idx = np.argmin(np.sum(archive_f, axis=1))
                return np.array([archive_f[best_idx]])
            return np.array([])
        
        # Create a list of active points for refinement
        active_x = [archive_x[i] for i in non_dom_indices]
        active_f = [archive_f[i] for i in non_dom_indices]
        
        # Distribute remaining budget
        rem_budget = self.budget - evals_used
        if rem_budget > 0 and active_x:
            steps_per_point = max(1, rem_budget // len(active_x))
            
            for i in range(len(active_x)):
                if evals_used >= self.budget or time.time() > deadline:
                    break
                
                current_x = active_x[i].copy()
                current_f = active_f[i].copy()
                
                for _ in range(steps_per_point):
                    if evals_used >= self.budget or time.time() > deadline:
                        break
                    
                    # Random local search step
                    j = rng.integers(0, self.dim)
                    step_size = 0.05 * (hi[j] - lo[j])
                    if j >= 2:
                        step_size *= 0.1 # Smaller steps for positioning vars
                    
                    delta = rng.standard_normal() * step_size
                    new_x = current_x.copy()
                    new_x[j] += delta
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    evals_used += 1
                    
                    # Accept if dominates
                    if np.all(new_f <= current_f) and np.any(new_f < current_f):
                        current_x = new_x
                        current_f = new_f
                    else:
                        # If not dominating, check if it's better in sum (to converge)
                        # But this might destroy diversity. 
                        # Let's only accept if it dominates.
                        pass
                        
                active_x[i] = current_x
                active_f[i] = current_f
        
        # Final non-dominated filter
        final_f = []
        for i in range(len(active_f)):
            is_dom = False
            for j in range(len(active_f)):
                if i == j: continue
                if np.all(active_f[j] <= active_f[i]) and np.any(active_f[j] < active_f[i]):
                    is_dom = True
                    break
            if not is_dom:
                final_f.append(active_f[i])
        
        if not final_f:
             # Fallback
             return np.array([active_f[0]])
             
        return np.array(final_f)

# EVOLVE-BLOCK-END
