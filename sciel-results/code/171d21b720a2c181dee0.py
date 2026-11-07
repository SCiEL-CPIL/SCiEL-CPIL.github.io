# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-based decomposition with large-step local Nelder-Mead simplex per cell."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []
        eval_count = [0]

        def eval_point(x):
            if eval_count[0] >= budget:
                raise StopIteration
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count[0] += 1
            return f

        def nelder_mead(func, x0, max_evals, lo, hi):
            # func takes x and returns scalar
            # x0 is initial point
            # max_evals is the number of evaluations allowed
            
            # Initial simplex
            simplex = [x0.copy()]
            # Large step size to allow global reach within the local search
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 * (hi[i] - lo[i]) if hi[i] > lo[i] else 0.1
                # Alternate signs to cover both directions? 
                # NM usually expands in one direction. 
                # Let's just use positive step for simplicity, or random?
                # Deterministic: positive step.
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            for x in simplex:
                f_values.append(func(x))
                
            # Sort by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Loop until we run out of evaluations
            while eval_count[0] < max_evals:
                # Check if we are close to budget
                if eval_count[0] >= max_evals:
                    break
                
                # Centroid of best points (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                
                # We need to evaluate xr. 
                # But we must ensure we don't exceed max_evals.
                # The loop condition checks before the iteration, but we might do multiple evals in one iteration.
                # Let's just call func and let eval_point raise StopIteration if needed, 
                # but we catch it? No, we want to stop cleanly.
                
                # Check if we have budget for at least 1 eval
                if eval_count[0] >= max_evals:
                    break
                
                fr = func(xr)
                
                if fr < f_values[0]:
                    # Expansion
                    if eval_count[0] >= max_evals:
                        break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        if eval_count[0] >= max_evals:
                            break
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, lo, hi)
                        fc = func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                if eval_count[0] >= max_evals:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lo, hi)
                                f_values[i] = func(simplex[i])
                    else:
                        # Inside contraction
                        if eval_count[0] >= max_evals:
                            break
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lo, hi)
                        fc = func(xc)
                        if fc <= f_values[-1]:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                if eval_count[0] >= max_evals:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lo, hi)
                                f_values[i] = func(simplex[i])
                    
                # Re-sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
            return simplex[0], f_values[0]

        # Grid
        grid_size = 2
        n_cells = grid_size ** dim
        
        # Weights
        weights = np.linspace(0, 1, n_cells)
        
        # Cell centers
        centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        center = lo + span * np.array([i + 0.5, j + 0.5, k + 0.5, l + 0.5]) / grid_size
                        centers.append(center)
        
        # Allocate budget
        # 16 cells. 
        # 200 / 16 = 12.5.
        # Give each cell 12 evaluations.
        
        for idx in range(n_cells):
            if eval_count[0] >= budget:
                break
            
            x0 = centers[idx]
            w = weights[idx]
            
            def scalar_obj(x, w=w):
                f = eval_point(x)
                return w * f[0] + (1 - w) * f[1]
            
            maxfev_local = min(12, budget - eval_count[0])
            
            try:
                nelder_mead(scalar_obj, x0, maxfev_local, lo, hi)
            except (StopIteration, Exception):
                pass

        return all_X
# EVOLVE-BLOCK-END
