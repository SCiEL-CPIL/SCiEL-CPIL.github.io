"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using Multi-start Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        mid = (lo + hi) / 2.0
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        if budget < 10:
            # Fallback for extremely small budgets
            X = lo + range_ * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Phase 1: Coarse Exploration via Latin Hypercube Sampling (LHS)
        # We want a diverse initial set to cover the Pareto front.
        # Let's use about 20% of the budget for initialization, but at least 10 points.
        # LHS ensures coverage of the search space.
        n_init = max(10, min(40, budget // 5))
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_init bins, pick one random point in each, then shuffle.
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, n_init + 1)
            # Pick a random point in each bin
            points_in_bin = np.random.uniform(bins[:-1], bins[1:])
            # Shuffle to break correlation between dimensions
            np.random.shuffle(points_in_bin)
            lhs_points[:, d] = points_in_bin
            
        X_init = lo + range_ * lhs_points
        F_init = np.array([problem(x) for x in X_init])
        
        used = n_init
        all_points = list(zip(X_init, F_init))
        
        # Maintain a set of "local optima" candidates
        # We will refine the best ones.
        # Sort initial points by a proxy for quality: distance to ideal point or simply keep them all.
        # For local search, we need a starting point.
        # Let's select a subset of the most promising initial points to refine.
        # How to select? 
        # 1. Non-dominated sorting (expensive to implement correctly from scratch quickly, but standard)
        # 2. Or simply pick points that are far from each other in objective space?
        # 3. Or just pick the first K?
        # Given the budget is small (200), let's pick a diverse set of local optima.
        # Let's pick the top 10 based on a simple scalarization (e.g. sum of normalized objectives) 
        # or just random selection from the non-dominated set if we can compute it cheaply.
        
        # Let's implement a quick non-dominated check for the initial set.
        # With n_init <= 40, O(n^2 * n_obj) is fine.
        def is_dominated(f, F_set):
            # Returns True if f is dominated by any in F_set
            # f: (n_obj,)
            # F_set: (n, n_obj)
            # f is dominated if exists g in F_set such that g <= f and g != f
            # g <= f means all components smaller or equal
            # g != f means at least one strictly smaller
            # Vectorized:
            # diff = F_set - f
            # dominated = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
            # return np.any(dominated)
            diff = F_set - f
            return np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))

        # Identify non-dominated points in F_init
        nd_mask = np.zeros(n_init, dtype=bool)
        for i in range(n_init):
            if not is_dominated(F_init[i], F_init):
                nd_mask[i] = True
                
        nd_indices = np.where(nd_mask)[0]
        
        # If we have few non-dominated points, we might have missed some good regions.
        # But for LHS, we should have several.
        # We want to run local search on a subset of these.
        # Let's say we run local search on up to 5 of the most "spread out" non-dominated points.
        # Or just the first 5.
        # To maximize hypervolume, we want to refine points that lead to distinct local optima.
        
        # Strategy: Allocate remaining budget to local search on a few promising seeds.
        # Let's pick 5 seeds from the non-dominated set.
        # If nd_indices is empty (shouldn't happen), pick random.
        if len(nd_indices) == 0:
            seeds = np.random.choice(n_init, size=min(5, n_init), replace=False)
        else:
            # Pick 5 from nd_indices. To ensure diversity, maybe pick evenly spaced indices?
            # Or just random.
            k = min(5, len(nd_indices))
            # Simple random selection from non-dominated
            seeds = np.random.choice(nd_indices, size=k, replace=False)
            
        # Local Search Algorithm: Nelder-Mead
        # We need to be careful with the number of function evaluations per local search.
        # Nelder-Mead on 12 dims can be expensive.
        # Standard NM: simplex of size dim+1 = 13 points.
        # Each iteration evaluates 1-2 points.
        # Max iterations per local search: (budget - used) / k / 10?
        
        remaining_budget = budget - used
        max_evals_per_ls = max(10, remaining_budget // (len(seeds) * 2))
        
        # We will run local search on each seed.
        # We need a scalar objective for NM.
        # Use Chebyshev distance to the current ideal point (or a fixed reference).
        # Reference: [3.0, 5.0, 7.0] is the hypervolume reference.
        # But for minimization, we want to minimize distance to the "ideal" corner.
        # Let's use the current best known ideal point (min of F_init) as the target.
        ideal_point = np.min(F_init, axis=0)
        
        # To ensure we explore the front, we might want to use different targets or just minimize a norm.
        # Let's minimize the sum of squared normalized deviations from the ideal point.
        # Or simply the L-infinity norm (Chebyshev) which aligns better with hypervolume near the ideal.
        
        def scalar_objective(x):
            fx = problem(x)
            # Clip to bounds is done by problem? "problem(x) clips... and returns"
            # So x can be outside? No, problem clips x.
            # But NM might generate points outside bounds.
            # We should clip x before passing to problem?
            # The contract says: "problem(x) clips a finite length-12 vector to the bounds"
            # So we can pass unclipped x.
            
            # Update global ideal point? 
            # If we update it, the objective function changes, which is bad for NM convergence.
            # Better to use a fixed target for each local search run, or the initial ideal point.
            # Let's use the initial ideal point for stability.
            
            # Normalize objectives?
            # Ranges are unknown. Let's use raw Chebyshev distance.
            dist = np.max(np.abs(fx - ideal_point))
            return dist, fx

        # We need to track evaluations.
        # NM implementation:
        
        def nelder_mead(start_x, max_evals, current_used, all_points):
            # start_x: (dim,)
            # Returns best_x, best_f, new_used, new_all_points
            
            # Initialize simplex
            # Start point
            x0 = start_x.copy()
            
            # Perturbation
            # If x0 is on a bound, perturb inwards.
            # Simple: add small random noise
            step = 0.05 * range_
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0 + np.random.uniform(-step, step)
                # Clip to bounds for the points themselves? 
                # Problem clips, so it's safe.
                # But for the geometry of NM, it's better to keep them within bounds if possible.
                simplex[i] = np.clip(simplex[i], lo, hi)
                
            # Evaluate initial simplex
            f_simplex = np.zeros((dim + 1, n_obj))
            local_used = 0
            local_points = []
            
            for i in range(dim + 1):
                if current_used + local_used >= budget:
                    break
                fx = problem(simplex[i])
                f_simplex[i] = fx
                local_used += 1
                local_points.append((simplex[i], fx))
                
                # Check time
                if time.time() > deadline:
                    break
                    
            if local_used == 0:
                return x0, f_simplex[0] if f_simplex[0].any() else np.zeros(n_obj), current_used, all_points

            # Sort simplex by scalar objective (Chebyshev to ideal_point)
            def get_scalar(f):
                return np.max(np.abs(f - ideal_point))
            
            scalars = np.array([get_scalar(f) for f in f_simplex])
            order = np.argsort(scalars)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            scalars = scalars[order]
            
            # Main Loop
            alpha = 1.0 # Reflection
            gamma = 2.0 # Expansion
            rho = 0.5   # Contraction
            sigma = 0.5 # Shrink
            
            # Convergence criterion
            tol = 1e-6
            max_iter = max_evals - local_used
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if current_used + local_used >= budget:
                    break
                    
                # Check convergence
                if np.ptp(scalars) < tol:
                    break
                    
                # 1. Reflection
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                s_worst = scalars[-1]
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                
                fx_r = problem(x_r)
                local_used += 1
                local_points.append((x_r, fx_r))
                s_r = get_scalar(fx_r)
                
                if s_r < scalars[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    fx_e = problem(x_e)
                    local_used += 1
                    local_points.append((x_e, fx_e))
                    s_e = get_scalar(fx_e)
                    
                    if s_e < s_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = fx_e
                        scalars[-1] = s_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = fx_r
                        scalars[-1] = s_r
                        
                    # Re-sort
                    order = np.argsort(scalars)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
                    scalars = scalars[order]
                    
                elif s_r < scalars[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = fx_r
                    scalars[-1] = s_r
                    # Re-sort
                    order = np.argsort(scalars)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
                    scalars = scalars[order]
                    
                else:
                    # Contraction
                    if s_r < s_worst:
                        # Outside Contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside Contraction
                        x_c = centroid - rho * (x_worst - centroid)
                        
                    x_c = np.clip(x_c, lo, hi)
                    fx_c = problem(x_c)
                    local_used += 1
                    local_points.append((x_c, fx_c))
                    s_c = get_scalar(fx_c)
                    
                    if s_c < min(s_r, s_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = fx_c
                        scalars[-1] = s_c
                        # Re-sort
                        order = np.argsort(scalars)
                        simplex = simplex[order]
                        f_simplex = f_simplex[order]
                        scalars = scalars[order]
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        x_best = simplex[0]
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fx_i = problem(simplex[i])
                            local_used += 1
                            local_points.append((simplex[i], fx_i))
                            f_simplex[i] = fx_i
                            scalars[i] = get_scalar(fx_i)
                            
                            if time.time() > deadline or current_used + local_used >= budget:
                                break
                                
                        if time.time() > deadline or current_used + local_used >= budget:
                            break
                            
                        # Re-sort
                        order = np.argsort(scalars)
                        simplex = simplex[order]
                        f_simplex = f_simplex[order]
                        scalars = scalars[order]

            # Return the best point found in this local search
            best_idx = 0 # simplex[0] is best
            best_x = simplex[0]
            best_f = f_simplex[0]
            
            return best_x, best_f, current_used + local_used, all_points + local_points

        # Run local search on selected seeds
        for seed_idx in seeds:
            if time.time() > deadline:
                break
            if used >= budget:
                break
                
            start_x = X_init[seed_idx]
            
            # Estimate max evals for this specific local search
            # We want to leave some budget for potential restarts or just use what's left.
            # Let's allocate remaining_budget / len(seeds) to each.
            alloc = (budget - used) // (len(seeds) - np.where(seeds == seed_idx)[0][0])
            alloc = max(10, min(alloc, budget - used))
            
            best_x, best_f, new_used, new_points = nelder_mead(start_x, alloc, used, all_points)
            
            if new_used > used:
                all_points = new_points
                used = new_used
                
                # Update global ideal point if this point is better
                ideal_point = np.minimum(ideal_point, best_f)
                
        # If we have budget left, do a few random restarts or refine the best point further?
        # Given the small budget, we've likely spent it all.
        # The all_points list contains all evaluated points.
        # The return value is ignored, but we must return something.
        # Let's return the last evaluated point or the best one.
        
        if all_points:
            # Return the last point evaluated
            return all_points[-1][0]
        else:
            return np.zeros(dim)

# EVOLVE-BLOCK-END
