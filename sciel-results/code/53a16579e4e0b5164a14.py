"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead Ensemble for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Multi-start Nelder-Mead parameters
        n_starts = 5
        # Initial step size for simplex expansion
        # Domain is [-5, 5], minimizer in [-4, 4].
        # A step of 0.5 covers a good local area.
        initial_step = 0.5
        
        # Nelder-Mead parameters
        # alpha, gamma, rho are standard reflection, expansion, contraction
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5  # Shrinkage factor
        
        # State for each simplex
        # simplex: (n+1, n) array of points
        # values: (n+1,) array of function values
        simplexes = []
        values = []
        
        # Generate Latin Hypercube points for initial starts
        # For dim=5, n_starts=5, this gives one point per dimension bin.
        # LHS: for each dimension, partition [0,1] into n_starts bins, pick one random point in each bin, permute.
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, n_starts + 1)
            # Pick one random point in each bin
            points_in_bin = [np.random.uniform(bins[i], bins[i+1]) for i in range(n_starts)]
            # Shuffle to ensure decorrelation across dimensions
            np.random.shuffle(points_in_bin)
            lhs_points[:, d] = points_in_bin
        
        # Map from [0,1]^dim to [-5,5]^dim
        # We want to ensure we cover the box well.
        # Center is 0, half-width is 5.
        # x = -5 + 10 * lhs
        start_points = -5.0 + 10.0 * lhs_points
        
        # Initialize simplexes
        for i in range(n_starts):
            center = start_points[i]
            # Create initial simplex
            # Vertex 0 is the center
            # Other vertices are center +/- step in each dimension
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for j in range(dim):
                simplex[j+1] = center.copy()
                simplex[j+1][j] += initial_step
            
            # Evaluate initial simplex
            vals = np.zeros(dim + 1)
            for k in range(dim + 1):
                x = np.clip(simplex[k], -5, 5)
                vals[k] = func(x)
                budget -= 1
                if budget <= 0:
                    break
                if time.time() > deadline:
                    return simplex[0], vals[0] # Early exit safety
            
            if budget <= 0:
                break
                
            simplexes.append(simplex)
            values.append(vals)
            
            if time.time() > deadline:
                break

        if not simplexes:
             # Fallback if even initialization fails
             x0 = np.zeros(dim)
             f0 = func(x0)
             return x0, f0

        # Track global best
        all_vals = np.concatenate(values)
        global_best_idx = np.argmin(all_vals)
        # Map back to simplex index and vertex index
        s_idx = global_best_idx // (dim + 1)
        v_idx = global_best_idx % (dim + 1)
        best_x = np.clip(simplexes[s_idx][v_idx], -5, 5)
        best_f = values[s_idx][v_idx]
        
        # Main Loop: Round-robin over simplexes
        # We use a fixed number of iterations per simplex or until budget/time runs out.
        # Since NM converges fast, we can run many iterations.
        # Let's run until budget or time is exhausted.
        
        active_simplexes = list(range(len(simplexes)))
        
        while budget > 0 and time.time() < deadline:
            # Iterate over active simplexes
            next_active = []
            for s_idx in active_simplexes:
                if budget <= 0 or time.time() >= deadline:
                    break
                
                simplex = simplexes[s_idx]
                vals = values[s_idx]
                
                # Sort vertices by value
                sort_idx = np.argsort(vals)
                simplex = simplex[sort_idx]
                vals = vals[sort_idx]
                
                # Check convergence
                # If the difference between best and worst is small, and step size is small, stop this simplex
                val_range = vals[-1] - vals[0]
                if val_range < 1e-10:
                    continue # Converged, skip for now (could restart, but let's keep it simple)
                
                # Identify worst, best, second best
                x_worst = simplex[-1]
                f_worst = vals[-1]
                x_best = simplex[0]
                f_best = vals[0]
                x_second = simplex[1]
                
                # Centroid of all but worst
                centroid = np.sum(simplex[:-1], axis=0) / dim
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                budget -= 1
                
                if f_ref < vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    budget -= 1
                    
                    if f_exp < f_ref:
                        new_x = x_exp
                        new_f = f_exp
                    else:
                        new_x = x_ref
                        new_f = f_ref
                elif f_ref < vals[-2]:
                    # Accept reflection
                    new_x = x_ref
                    new_f = f_ref
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    budget -= 1
                    
                    if f_con < f_ref:
                        new_x = x_con
                        new_f = f_con
                    else:
                        # Shrinkage
                        # Shrink all points except best towards best
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        # Re-evaluate shrunk points
                        for i in range(1, dim + 1):
                            vals[i] = func(simplex[i])
                            budget -= 1
                            if budget <= 0:
                                break
                            
                        if budget <= 0:
                            break
                        
                        # Update global best
                        local_best_idx = np.argmin(vals)
                        if vals[local_best_idx] < best_f:
                            best_f = vals[local_best_idx]
                            best_x = np.clip(simplex[local_best_idx], -5, 5)
                        
                        # Check if simplex is too small
                        if np.linalg.norm(simplex[0] - simplex[1]) < 1e-8:
                            continue # Mark as converged, don't add to next_active
                            
                        simplexes[s_idx] = simplex
                        values[s_idx] = vals
                        next_active.append(s_idx)
                        continue # Skip adding to next_active in this iteration? 
                        # Actually, shrinkage resets the simplex, so we should keep it active 
                        # but maybe limit consecutive shrinkages. For now, keep active.
                        # But we already updated simplexes[s_idx]. 
                        # Let's just add to next_active.
                        # Wait, the logic above 'continue's inside the loop, so it skips the append at the end.
                        # Let's restructure to be cleaner.
                        # I will handle the append logic outside the if/else block properly.
                        
                        # For this specific branch (shrinkage), we updated the simplex in place.
                        # We should add to next_active.
                        next_active.append(s_idx)
                        continue

                # Replace worst point
                simplex[-1] = new_x
                vals[-1] = new_f
                
                # Update global best
                local_best_idx = np.argmin(vals)
                if vals[local_best_idx] < best_f:
                    best_f = vals[local_best_idx]
                    best_x = np.clip(simplex[local_best_idx], -5, 5)
                
                # Check convergence again after update
                # If range is tiny, stop
                if (vals[-1] - vals[0]) < 1e-10:
                    continue
                    
                simplexes[s_idx] = simplex
                values[s_idx] = vals
                next_active.append(s_idx)
                
                if budget <= 0 or time.time() >= deadline:
                    break
            
            active_simplexes = next_active
            
            # If no active simplexes, we are done
            if not active_simplexes:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
