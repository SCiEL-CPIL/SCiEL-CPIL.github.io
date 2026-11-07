"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-free continuous multi-objective optimization for ZDT1.
    
    Strategy: 
    1. Select K target points along the Pareto front (x1 values).
    2. For each target, initialize x with x1=target and x[1:]=0.0.
    3. Use Nelder-Mead local search on x[1:] to minimize f2, keeping x[0] fixed.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _optimize_tail(self, problem, x_fixed, bounds, budget):
        """
        Optimizes x[1:] given x[0] fixed using Nelder-Mead.
        Returns best x, best f, evals used.
        """
        dim_tail = self.dim - 1
        if dim_tail == 0:
            x = np.zeros(self.dim)
            x[0] = x_fixed
            f = problem(x)
            return x, f, 1

        lo_tail = np.array([b[0] for b in bounds[1:]])
        hi_tail = np.array([b[1] for b in bounds[1:]])
        
        # Initial y (tail)
        y = np.zeros(dim_tail)
        y = np.clip(y, lo_tail, hi_tail)
        
        def eval_y(y_vec):
            x = np.zeros(self.dim)
            x[0] = x_fixed
            x[1:] = y_vec
            return problem(x)
            
        f_init = eval_y(y)
        used = 1
        
        if budget <= 1:
            x_best = np.zeros(self.dim)
            x_best[0] = x_fixed
            x_best[1:] = y
            return x_best, f_init, used

        # Simplex vertices: list of (y, f2)
        simplex = [(y.copy(), f_init[1])]
        
        # Generate initial simplex
        for i in range(dim_tail):
            if used >= budget:
                break
            y_new = y.copy()
            # Step size: 10% of range
            range_i = hi_tail[i] - lo_tail[i]
            step = 0.1 * range_i if range_i > 0 else 0.1
            y_new[i] += step
            y_new = np.clip(y_new, lo_tail, hi_tail)
            
            f_new = eval_y(y_new)
            used += 1
            simplex.append((y_new.copy(), f_new[1]))
            
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main loop
        while used < budget and len(simplex) > 1:
            # Sort simplex by f2
            simplex.sort(key=lambda kv: kv[1])
            
            y_worst = simplex[-1][0]
            f_worst = simplex[-1][1]
            y_best = simplex[0][0]
            f_best = simplex[0][1]
            
            # Centroid of all but worst
            centroid = np.mean([v[0] for v in simplex[:-1]], axis=0)
            
            # Reflection
            y_r = centroid + alpha * (centroid - y_worst)
            y_r = np.clip(y_r, lo_tail, hi_tail)
            f_r = eval_y(y_r)[1]
            used += 1
            
            if f_r < simplex[1][1] and f_r >= simplex[0][1]:
                # Replace worst with reflection
                simplex[-1] = (y_r.copy(), f_r)
                continue
            elif f_r < simplex[0][1]:
                # Expansion
                y_e = centroid + gamma * (centroid - y_worst)
                y_e = np.clip(y_e, lo_tail, hi_tail)
                f_e = eval_y(y_e)[1]
                used += 1
                if f_e < f_r:
                    simplex[-1] = (y_e.copy(), f_e)
                else:
                    simplex[-1] = (y_r.copy(), f_r)
                continue
            else:
                # Contraction
                if f_r < f_worst:
                    # Inside contraction
                    y_c = centroid + rho * (centroid - y_worst)
                else:
                    # Outside contraction
                    y_c = centroid - rho * (centroid - y_worst)
                    
                y_c = np.clip(y_c, lo_tail, hi_tail)
                f_c = eval_y(y_c)[1]
                used += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = (y_c.copy(), f_c)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if used >= budget:
                            break
                        simplex[i][0] = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                        simplex[i][0] = np.clip(simplex[i][0], lo_tail, hi_tail)
                        f_shrink = eval_y(simplex[i][0])[1]
                        used += 1
                        simplex[i] = (simplex[i][0].copy(), f_shrink)
                        
        # Find best in simplex
        if not simplex:
             # Fallback
             x_best = np.zeros(self.dim)
             x_best[0] = x_fixed
             x_best[1:] = y
             f_best_vec = eval_y(y)
             return x_best, f_best_vec, used + 1

        best_idx = np.argmin([v[1] for v in simplex])
        best_y = simplex[best_idx][0]
        
        # Reconstruct x and get full f vector
        x_best = np.zeros(self.dim)
        x_best[0] = x_fixed
        x_best[1:] = best_y
        
        # Ensure we have the exact f vector from the problem for the best point
        # We might have evaluated it already, but to be safe and consistent with recording:
        # If we didn't evaluate this specific point as the 'best' in the last step, we should.
        # However, NM keeps the value in the simplex.
        # But problem() might return slightly different due to float issues? Unlikely.
        # To be safe, let's do one final eval if we have budget.
        if used < self.budget:
            f_best_vec = eval_y(best_y)
            used += 1
        else:
            # Use the stored f2 and construct f1
            f_best_vec = np.array([x_fixed, simplex[best_idx][1]])
            
        return x_best, f_best_vec, used

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine number of targets
        # NM init cost is roughly dim (for simplex) + 1 (init)
        # Let's use 3 targets to allow for decent NM iterations.
        n_targets = 3
        
        # Check if budget is sufficient for NM init
        # Approx cost per target: 1 (init) + (dim-1) (simplex) + iterations
        # If budget is very low, reduce targets.
        min_cost_per_target = self.dim + 1
        if self.budget < n_targets * min_cost_per_target:
            n_targets = max(1, self.budget // min_cost_per_target)
            
        # Targets for x1 (which is f1)
        # Spread evenly, avoiding 0 to prevent potential issues if any
        x1_targets = np.linspace(0.1, 0.9, n_targets)
        
        archive = []
        used = 0
        
        for t in x1_targets:
            if used >= self.budget:
                break
                
            # Remaining budget for this target
            budget_left = self.budget - used
            
            # Optimize tail
            x_best, f_best, evals_used = self._optimize_tail(problem, t, bounds, budget_left)
            
            archive.append({'x': x_best.copy(), 'f': f_best.copy()})
            used += evals_used
            
        return np.array([a['x'] for a in archive])
# EVOLVE-BLOCK-END
