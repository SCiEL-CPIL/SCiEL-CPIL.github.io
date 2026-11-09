import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses diverse initial points to quickly locate the basin of attraction
    for the BBOB f6 (Attractive Sector) function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget

        # Helper to evaluate and track budget
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Define diverse initial points for restarts
        # We use 5 restarts.
        # The points are chosen to spread across the domain [-5, 5]^d
        dim = self.dim
        # Scale factor to ensure points are within [-5, 5]
        # Using a set of orthogonal-ish vectors
        base_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            -np.ones(dim) * 1.0,
            np.array([1, -1, 1, -1, 1][:dim] if dim >= 5 else [1, -1, 1, -1, 1][:dim]),
            np.array([-1, 1, -1, 1, -1][:dim] if dim >= 5 else [-1, 1, -1, 1, -1][:dim]),
        ]
        
        # Scale points to be within bounds, roughly centered
        # For f6, the optimum is shifted. Spreading is good.
        # Let's scale them to have a magnitude of ~3.0 to cover the space
        scaled_points = []
        for p in base_points:
            # Normalize and scale
            norm = np.linalg.norm(p)
            if norm > 0:
                p = p / norm * 3.0
            else:
                p = np.zeros(dim)
            # Clip to bounds
            p = np.clip(p, -5.0, 5.0)
            scaled_points.append(p)

        num_restarts = 5
        budget_per_restart = max(1, (max_calls - 1) // num_restarts)
        
        # We reserve 1 call for the very first point evaluation if needed, 
        # but NM simplex requires d+1 points.
        # Total calls for one NM run: (d+1) initial + iterations.
        
        for r in range(num_restarts):
            if calls_made >= max_calls:
                break
            if time.time() > deadline:
                break

            # Initialize simplex around the start point
            start_pt = scaled_points[r % len(scaled_points)]
            
            # Simplex initialization
            # Vertex 0: start_pt
            # Vertices 1..d: start_pt + delta * e_i
            delta = 0.5  # Initial step size
            simplex = [start_pt.copy()]
            for i in range(dim):
                v = start_pt.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                fv = eval_func(v)
                fvals.append(fv)
            
            if len(fvals) < dim + 1:
                continue

            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Local state for NM
            # We need to track the simplex and its function values
            # simplex is list of arrays, fvals is list of floats
            
            # Sort simplex by function value
            # We will maintain simplex sorted such that simplex[0] is best
            
            def sort_simplex():
                nonlocal simplex, fvals
                # Create indices sorted by fvals
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
            
            sort_simplex()
            
            iterations = 0
            max_iters = budget_per_restart - (dim + 1)
            
            while iterations < max_iters:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                
                # Check for convergence/stagnation
                # If the best and worst are very close, stop
                if abs(fvals[0] - fvals[-1]) < 1e-8:
                    break
                
                # Calculate centroid of all points except the worst
                worst_idx = dim # The last one in sorted order
                centroid = np.mean(simplex[:worst_idx], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[worst_idx])
                xr = np.clip(xr, -5.0, 5.0)
                fr = eval_func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[worst_idx])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = eval_func(xe)
                    if fe < fr:
                        simplex[worst_idx] = xe
                        fvals[worst_idx] = fe
                    else:
                        simplex[worst_idx] = xr
                        fvals[worst_idx] = fr
                elif fr < fvals[worst_idx - 1]:
                    # Accept reflection
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
                else:
                    # Contraction
                    # Outside contraction if fr < worst, inside if fr >= worst
                    # Standard: If fr >= fvals[worst_idx-1] (second worst), we are in trouble.
                    # Let's use standard contraction logic.
                    
                    if fr < fvals[worst_idx]:
                        # Outside contraction
                        xc = centroid + (1 - alpha) * (centroid - simplex[worst_idx])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = eval_func(xc)
                        if fc < fr:
                            simplex[worst_idx] = xc
                            fvals[worst_idx] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = eval_func(simplex[i])
                            break # Restart NM if shrink happens too often? 
                                 # Or just continue. Shrink reduces size.
                    else:
                        # Inside contraction
                        xc = centroid + (1 - alpha) * (simplex[worst_idx] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = eval_func(xc)
                        if fc < fvals[worst_idx]:
                            simplex[worst_idx] = xc
                            fvals[worst_idx] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = eval_func(simplex[i])
                            break
                            
                sort_simplex()
                iterations += 1
                
        return best_x, best_f
