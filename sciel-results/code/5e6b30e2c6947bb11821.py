import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def time_ok():
            return time.time() < deadline and calls_used < self.budget

        # Initial global exploration phase
        # We divide the budget into a few initial restarts to find good basins
        # For dim=5, 5 restarts of 50 calls each = 250 calls.
        # Then we refine the best one with the remaining budget.
        
        n_initial = 5
        budget_per_initial = max(10, (self.budget // 2) // n_initial)
        
        candidates = []
        
        for i in range(n_initial):
            if not time_ok():
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            calls_used += 1
            
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
                
            candidates.append((fx0, x0.copy()))
            
            # Simple local refinement for this initial point
            # Use Nelder-Mead with a small budget
            x_curr = x0.copy()
            f_curr = fx0
            
            # Initialize simplex
            simplex = [x_curr]
            f_simplex = [f_curr]
            
            # Add other vertices
            for j in range(self.dim):
                v = x_curr.copy()
                # Heuristic step size: 10% of domain size if 0, else 5% of current value
                step = 0.5 if abs(v[j]) < 0.5 else 0.05 * abs(v[j])
                v[j] += step
                fv = func(v)
                calls_used += 1
                simplex.append(v)
                f_simplex.append(fv)
                
                if fv < f_curr:
                    f_curr = fv
                    x_curr = v.copy()
                    if fv < best_f:
                        best_f = fv
                        best_x = v.copy()

            # Run a few steps of Nelder-Mead on this candidate
            # Limit steps to avoid spending too much time on bad basins
            max_nm_steps = 20 
            for _ in range(max_nm_steps):
                if not time_ok():
                    break
                    
                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fxr = func(xr)
                calls_used += 1
                
                if fxr < f_simplex[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fxe = func(xe)
                    calls_used += 1
                    
                    if fxe < fxr:
                        simplex[-1] = xe
                        f_simplex[-1] = fxe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fxr
                elif fxr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fxr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fxc = func(xc)
                    calls_used += 1
                    
                    if fxc < fxr:
                        simplex[-1] = xc
                        f_simplex[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls_used += 1
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()
                            if not time_ok():
                                break
            
            # Update best from this candidate's final state
            min_idx = np.argmin(f_simplex)
            if f_simplex[min_idx] < best_f:
                best_f = f_simplex[min_idx]
                best_x = simplex[min_idx].copy()

        # Final refinement phase
        # Use remaining budget on the global best found so far
        if best_x is not None and time_ok():
            remaining_budget = self.budget - calls_used
            if remaining_budget > 0:
                x_curr = best_x.copy()
                f_curr = best_f
                
                # Initialize simplex around best_x
                simplex = [x_curr]
                f_simplex = [f_curr]
                
                for j in range(self.dim):
                    v = x_curr.copy()
                    step = 0.1 if abs(v[j]) < 0.5 else 0.05 * abs(v[j])
                    v[j] += step
                    fv = func(v)
                    calls_used += 1
                    simplex.append(v)
                    f_simplex.append(fv)
                    
                    if fv < f_curr:
                        f_curr = fv
                        x_curr = v.copy()
                        if fv < best_f:
                            best_f = fv
                            best_x = v.copy()

                # Run Nelder-Mead until budget or time runs out
                while time_ok() and calls_used < self.budget:
                    # Sort simplex
                    idx = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in idx]
                    f_simplex = [f_simplex[i] for i in idx]
                    
                    # Check convergence (simple heuristic)
                    if f_simplex[-1] - f_simplex[0] < 1e-10:
                        break
                        
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    fxr = func(xr)
                    calls_used += 1
                    
                    if fxr < f_simplex[0]:
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex[-1])
                        fxe = func(xe)
                        calls_used += 1
                        
                        if fxe < fxr:
                            simplex[-1] = xe
                            f_simplex[-1] = fxe
                        else:
                            simplex[-1] = xr
                            f_simplex[-1] = fxr
                    elif fxr < f_simplex[-2]:
                        simplex[-1] = xr
                        f_simplex[-1] = fxr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fxc = func(xc)
                        calls_used += 1
                        
                        if fxc < fxr:
                            simplex[-1] = xc
                            f_simplex[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_simplex[i] = func(simplex[i])
                                calls_used += 1
                                if f_simplex[i] < best_f:
                                    best_f = f_simplex[i]
                                    best_x = simplex[i].copy()
                                if not time_ok():
                                    break
                    
                    # Update global best
                    min_idx = np.argmin(f_simplex)
                    if f_simplex[min_idx] < best_f:
                        best_f = f_simplex[min_idx]
                        best_x = simplex[min_idx].copy()

        return best_x, best_f
