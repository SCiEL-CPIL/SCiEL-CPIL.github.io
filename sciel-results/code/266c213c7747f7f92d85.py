"""
Optimizer for Bent Cigar (f12) using a combination of:
1. Sobol global search to locate the basin
2. Nelder-Mead simplex method with adaptive step sizes for anisotropic refinement
3. Fallback coordinate descent if Nelder-Mead stalls
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Nelder-Mead with adaptive scaling ---
        # Initialize simplex around best point
        x0 = x0.copy()
        fx0 = best_fx
        
        # Create initial simplex
        simplex = []
        simplex.append(x0.copy())
        simplex.append(fx0)
        
        # Create other vertices with small perturbations
        for i in range(dim):
            vertex = x0.copy()
            # Use a moderate initial step
            step = 0.5
            vertex[i] += step
            vertex = np.clip(vertex, -5, 5)
            fv = wrapped_obj(vertex)
            simplex.append(vertex)
            simplex.append(fv)
        
        # Sort simplex by function value
        def sort_simplex(simplex):
            # Extract points and values
            points = [simplex[i] for i in range(0, len(simplex), 2)]
            values = [simplex[i] for i in range(1, len(simplex), 2)]
            # Sort by values
            sorted_idx = np.argsort(values)
            new_simplex = []
            for idx in sorted_idx:
                new_simplex.append(points[idx].copy())
                new_simplex.append(values[idx])
            return new_simplex
        
        simplex = sort_simplex(simplex)
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        no_improve_count = 0
        max_no_improve = 50
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # Extract current simplex
            points = [simplex[i] for i in range(0, len(simplex), 2)]
            values = [simplex[i] for i in range(1, len(simplex), 2)]
            
            best_val = values[0]
            worst_val = values[-1]
            centroid = np.mean(points[:-1], axis=0)
            
            # Check convergence
            if worst_val - best_val < 1e-10 * (1 + abs(best_val)):
                break
            
            # Reflection
            x_r = centroid + alpha * (centroid - points[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = wrapped_obj(x_r)
            
            if f_r < best_val:
                # Expansion
                x_e = centroid + gamma * (centroid - points[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = wrapped_obj(x_e)
                if f_e < f_r:
                    new_point = x_e
                    new_val = f_e
                else:
                    new_point = x_r
                    new_val = f_r
            elif f_r < worst_val:
                new_point = x_r
                new_val = f_r
            else:
                # Contraction
                if f_r < worst_val:
                    # Outside contraction
                    x_c = centroid + rho * (points[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_obj(x_c)
                    if f_c <= f_r:
                        new_point = x_c
                        new_val = f_c
                    else:
                        # Shrink
                        shrink_points = [points[0]]
                        shrink_vals = [values[0]]
                        for i in range(1, len(points)):
                            new_p = points[0] + sigma * (points[i] - points[0])
                            new_p = np.clip(new_p, -5, 5)
                            new_v = wrapped_obj(new_p)
                            shrink_points.append(new_p)
                            shrink_vals.append(new_v)
                        simplex = []
                        for i in range(len(shrink_points)):
                            simplex.append(shrink_points[i])
                            simplex.append(shrink_vals[i])
                        simplex = sort_simplex(simplex)
                        continue
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - points[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_obj(x_c)
                    if f_c <= worst_val:
                        new_point = x_c
                        new_val = f_c
                    else:
                        # Shrink
                        shrink_points = [points[0]]
                        shrink_vals = [values[0]]
                        for i in range(1, len(points)):
                            new_p = points[0] + sigma * (points[i] - points[0])
                            new_p = np.clip(new_p, -5, 5)
                            new_v = wrapped_obj(new_p)
                            shrink_points.append(new_p)
                            shrink_vals.append(new_v)
                        simplex = []
                        for i in range(len(shrink_points)):
                            simplex.append(shrink_points[i])
                            simplex.append(shrink_vals[i])
                        simplex = sort_simplex(simplex)
                        continue
            
            # Replace worst point
            simplex[-2] = new_point
            simplex[-1] = new_val
            simplex = sort_simplex(simplex)
            
            # Track improvement
            if new_val < best_val - 1e-12:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Try coordinate descent as fallback
                break
        
        # --- Phase 3: Coordinate Descent for final refinement ---
        if calls_used < budget and time.time() < deadline:
            current_x = best_x.copy()
            current_fx = best_fx
            
            # Initialize step sizes
            h = np.full(dim, 0.1)
            
            no_improve_count = 0
            max_no_improve = 30
            
            while calls_used < budget:
                if time.time() >= deadline:
                    break
                
                improved = False
                
                for i in range(dim):
                    if time.time() >= deadline or calls_used >= budget:
                        break
                    
                    # Try positive and negative steps
                    x_plus = current_x.copy()
                    x_plus[i] += h[i]
                    x_plus = np.clip(x_plus, -5, 5)
                    
                    x_minus = current_x.copy()
                    x_minus[i] -= h[i]
                    x_minus = np.clip(x_minus, -5, 5)
                    
                    # Evaluate
                    fx_plus = wrapped_obj(x_plus)
                    fx_minus = wrapped_obj(x_minus)
                    
                    # Determine best move
                    if fx_plus < current_fx:
                        current_x = x_plus
                        current_fx = fx_plus
                        improved = True
                        h[i] = min(h[i] * 1.1, 5.0)
                    elif fx_minus < current_fx:
                        current_x = x_minus
                        current_fx = fx_minus
                        improved = True
                        h[i] = min(h[i] * 1.1, 5.0)
                    else:
                        # Failure: Shrink step size for this dimension
                        h[i] *= 0.5
                        if h[i] < 1e-8:
                            h[i] = 0.0
                            continue
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    if np.min(h[h>0]) < 1e-6:
                        break
                    h *= 1.5
                    no_improve_count = 0
        
        return best_x, best_fx
