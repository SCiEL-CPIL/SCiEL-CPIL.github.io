import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        # Phase 1: Grid search to find multiple basins
        # For 5D, use a 3^5 = 243 point grid
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([
                                -4 + 4 * i / 2,
                                -4 + 4 * j / 2,
                                -4 + 4 * k / 2,
                                -4 + 4 * l / 2,
                                -4 + 4 * m / 2
                            ])
                            grid_points.append(x)

        # Evaluate grid points
        grid_fvals = []
        calls_used = 0
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            f = func(x)
            grid_fvals.append(f)
            calls_used += 1
            if time.time() > deadline:
                break

        if not grid_fvals:
            # Fallback: just return origin
            x0 = np.zeros(dim)
            f0 = func(x0)
            return x0, f0

        grid_fvals = np.array(grid_fvals)
        grid_points = np.array(grid_points)

        # Find the best grid point
        best_idx = np.argmin(grid_fvals)
        best_x = grid_points[best_idx].copy()
        best_f = grid_fvals[best_idx]

        # Select top diverse points for local optimization
        # Sort by fitness and pick diverse ones
        sorted_indices = np.argsort(grid_fvals)
        
        # Pick up to 8 diverse starting points
        selected_points = []
        selected_fvals = []
        min_dist = 1.0  # Minimum distance between selected points
        
        for idx in sorted_indices:
            if len(selected_points) >= 8:
                break
            x = grid_points[idx]
            f = grid_fvals[idx]
            
            # Check if this point is sufficiently different from already selected
            is_diverse = True
            for sp in selected_points:
                if np.linalg.norm(x - sp) < min_dist:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_points.append(x)
                selected_fvals.append(f)

        # If we couldn't find enough diverse points, just take the best ones
        if len(selected_points) < 8:
            for idx in sorted_indices:
                if len(selected_points) >= 8:
                    break
                x = grid_points[idx]
                f = grid_fvals[idx]
                if x not in selected_points:
                    selected_points.append(x)
                    selected_fvals.append(f)

        # Phase 2: Nelder-Mead local optimization from each selected point
        for start_idx, (x_start, f_start) in enumerate(zip(selected_points, selected_fvals)):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 50
            
            # Initialize simplex
            simplex = [x_start.copy()]
            f_simplex = [f_start]
            
            # Create initial simplex
            step = 0.5
            for d in range(dim):
                x_new = x_start.copy()
                x_new[d] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_used += 1
                simplex.append(x_new)
                f_simplex.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                if calls_used >= budget or time.time() > deadline:
                    break
            
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Nelder-Mead iterations
            for iteration in range(max_iter):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                
                if f_ref < f_simplex[-2]:
                    if f_ref < f_simplex[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        calls_used += 1
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_simplex[-1] = f_exp
                            if f_exp < best_f:
                                best_f = f_exp
                                best_x = x_exp.copy()
                        else:
                            simplex[-1] = x_ref
                            f_simplex[-1] = f_ref
                            if f_ref < best_f:
                                best_f = f_ref
                                best_x = x_ref.copy()
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                            simplex[k] = np.clip(simplex[k], -5, 5)
                            f_simplex[k] = func(simplex[k])
                            calls_used += 1
                            
                            if f_simplex[k] < best_f:
                                best_f = f_simplex[k]
                                best_x = simplex[k].copy()
                            
                            if calls_used >= budget or time.time() > deadline:
                                break
                        
                        if calls_used >= budget or time.time() > deadline:
                            break
                
                # Check convergence
                if np.std(f_simplex) < 1e-12:
                    break

        return best_x, best_f
