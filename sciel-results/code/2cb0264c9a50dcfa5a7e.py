"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n_starts = 5
        dim = self.dim
        budget = self.budget

        # Generate space-filling initial points using Latin Hypercube sampling
        # For 5 starts in 5D, we can use a simple approach:
        # Divide each dimension into 5 intervals and pick one point per interval
        # with random offsets within each interval.
        
        # Actually, let's use a simpler deterministic + random approach for diversity:
        # Mix of center, corners, and random points
        
        # Let's use Latin Hypercube properly:
        # For n_starts points in dim dimensions:
        # For each dimension, divide [-5,5] into n_starts equal intervals
        # Assign each start a unique interval per dimension via permutation
        
        lhs_points = []
        for i in range(n_starts):
            point = np.zeros(dim)
            for d in range(dim):
                # Interval for this dimension and start
                interval_start = -5 + (10.0 / n_starts) * i
                interval_end = -5 + (10.0 / n_starts) * (i + 1)
                point[d] = rng.uniform(interval_start, interval_end)
            lhs_points.append(point)
        
        # Actually, proper LHS: for each dimension, permute the intervals
        lhs_points = []
        for d in range(dim):
            perm = rng.permutation(n_starts)
            for i in range(n_starts):
                if i == 0:
                    point = np.zeros(dim)
                interval_idx = perm[i]
                interval_start = -5 + (10.0 / n_starts) * interval_idx
                interval_end = -5 + (10.0 / n_starts) * (interval_idx + 1)
                point[d] = rng.uniform(interval_start, interval_end)
                if i == 0:
                    lhs_points.append(point)
        
        # Wait, the above is wrong. Let me redo LHS properly.
        # LHS: For n points in d dimensions:
        # For each dimension j, generate n uniform values in n strata
        # Then assign each point to get one value from each dimension
        
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            # Generate n_starts strata, pick one random point in each
            strata = np.array([
                rng.uniform(-5 + 10.0 * i / n_starts, -5 + 10.0 * (i + 1) / n_starts)
                for i in range(n_starts)
            ])
            # Assign to points via permutation
            perm = rng.permutation(n_starts)
            for i in range(n_starts):
                lhs_points[i, d] = strata[perm[i]]
        
        # Initialize best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Budget allocation: we have budget total calls
        # Each Nelder-Mead simplex starts with dim+1 points
        # We'll allocate calls round-robin among the starts
        
        # Initialize simplexes
        simplex_starts = []
        for i in range(n_starts):
            x0 = lhs_points[i]
            fx0 = func(x0)
            calls_made += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            # Build initial simplex around x0
            simplex = [x0.copy()]
            simplex_fvals = [fx0]
            
            # Add dim more points, each differing from x0 in one coordinate
            for d in range(dim):
                step = 0.1 * (1 if x0[d] < 4 else -1)  # Small step, avoid boundary
                new_point = x0.copy()
                new_point[d] += step
                new_point = np.clip(new_point, -5, 5)
                if np.allclose(new_point, x0):
                    # Try other direction
                    new_point = x0.copy()
                    new_point[d] -= step
                    new_point = np.clip(new_point, -5, 5)
                fnew = func(new_point)
                calls_made += 1
                if fnew < best_f:
                    best_f = fnew
                    best_x = new_point.copy()
                simplex.append(new_point)
                simplex_fvals.append(fnew)
            
            simplex_starts.append({
                'simplex': simplex,
                'fvals': simplex_fvals,
                'active': True
            })
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Main loop: distribute remaining calls among active simplexes
        remaining = budget - calls_made
        
        while remaining > 0 and any(s['active'] for s in simplex_starts):
            # Round-robin: pick next active simplex
            for si, s in enumerate(simplex_starts):
                if remaining <= 0:
                    break
                if not s['active']:
                    continue
                
                simplex = s['simplex']
                fvals = s['fvals']
                dim = self.dim
                n = len(simplex)  # should be dim + 1
                
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                remaining -= 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < fvals[0]:
                    # Try expansion
                    if remaining <= 0:
                        break
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    remaining -= 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    
                    if f_exp < f_ref:
                        x_new, f_new = x_exp, f_exp
                    else:
                        x_new, f_new = x_ref, f_ref
                elif f_ref < fvals[-2]:
                    x_new, f_new = x_ref, f_ref
                else:
                    # Contraction
                    if remaining <= 0:
                        break
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    remaining -= 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    
                    if f_con < min(f_ref, f_worst):
                        x_new, f_new = x_con, f_con
                    else:
                        # Shrink
                        if remaining <= 0:
                            break
                        # Shrink all points except best
                        x_shrunk = []
                        f_shrunk = []
                        for i in range(1, n):
                            x_shr = simplex[0] + sigma * (simplex[i] - simplex[0])
                            x_shr = np.clip(x_shr, -5, 5)
                            f_shr = func(x_shr)
                            calls_made += 1
                            remaining -= 1
                            if f_shr < best_f:
                                best_f = f_shr
                                best_x = x_shr.copy()
                            x_shrunk.append(x_shr)
                            f_shrunk.append(f_shr)
                        simplex = [simplex[0]] + x_shrunk
                        fvals = [fvals[0]] + f_shrunk
                        s['simplex'] = simplex
                        s['fvals'] = fvals
                        continue
                
                # Replace worst point
                simplex[-1] = x_new
                fvals[-1] = f_new
                s['simplex'] = simplex
                s['fvals'] = fvals
                
                # Check convergence: if simplex is very small, deactivate
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.max(spread) < 1e-6:
                    s['active'] = False
        
        return best_x, best_f
# EVOLVE-BLOCK-END
