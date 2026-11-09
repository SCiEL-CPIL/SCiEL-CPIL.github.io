"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Fixed diverse starting points to cover the box
        # 1. Origin
        # 2. Corners (8 for dim=5? No, 2^5=32 is too many. Let's use a few strategic ones)
        # Let's use a small set of deterministic points:
        # Origin, +/-4 in each dimension (8 points), and maybe mid-box points.
        
        start_points = []
        start_points.append(np.zeros(dim))
        
        # +/- 4.0 in each dimension
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                start_points.append(p)
                
        # Maybe a few random-ish but deterministic points?
        # Let's add points with mixed signs
        rng = np.random.default_rng(123)
        for _ in range(5):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)

        # Deduplicate (not strictly necessary but good practice)
        unique_starts = []
        for p in start_points:
            is_dup = False
            for u in unique_starts:
                if np.allclose(p, u):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(p)
                
        starts = unique_starts
        n_starts = len(starts)
        
        # Budget allocation: give roughly equal budget to each start
        # But we need to handle the "best so far" tracking for AOCC.
        # AOCC depends on the sequence of best values.
        # If we run start 1, then start 2, the "best value after call i" is the min of all calls so far.
        # So we just need to keep a global best_f and global best_x.
        
        calls_made = 0
        best_f = np.inf
        best_x = None
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget_left):
            nonlocal calls_made, best_f, best_x
            
            if calls_made >= budget:
                return x0, best_f

            dim = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                fx = func(simplex[i])
                calls_made += 1
                f_values.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()
            
            if len(f_values) < dim + 1:
                return best_x, best_f

            # Sort simplex by function value
            simplex = [simplex[i] for i in np.argsort(f_values)]
            f_values = [f_values[i] for i in np.argsort(f_values)]
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls_made < budget:
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = func(x_ref)
                calls_made += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_values[0] <= f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    f_con = func(x_con)
                    calls_made += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= budget:
                                break
                            f_shrink = func(simplex[i])
                            calls_made += 1
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                            f_values[i] = f_shrink
                        
                        # Re-sort
                        sort_idx = np.argsort(f_values)
                        simplex = [simplex[i] for i in sort_idx]
                        f_values = [f_values[i] for i in sort_idx]
                        continue # Restart loop logic with new simplex
                
                # Re-sort simplex
                sort_idx = np.argsort(f_values)
                simplex = [simplex[i] for i in sort_idx]
                f_values = [f_values[i] for i in sort_idx]
                
                # Check convergence
                if max(f_values) - min(f_values) < 1e-10:
                    break
                    
            return best_x, best_f

        # Run multi-start
        for start in starts:
            if calls_made >= budget:
                break
            nelder_mead(func, start, budget - calls_made)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
