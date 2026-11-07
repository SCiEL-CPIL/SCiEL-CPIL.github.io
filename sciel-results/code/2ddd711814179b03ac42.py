"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based decomposition with local refinement for RE37 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Track all evaluated points and their objectives
        X_all = []
        F_all = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f
        
        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget
        
        # Phase 1: Coarse grid evaluation
        # For 4D with budget 200, use a 3x3x3x3 grid = 81 points
        # This gives systematic coverage of the domain
        grid_size = 3
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1),
                        ])
                        grid_points.append(x)
        
        # Evaluate grid points
        for x in grid_points:
            if not time_ok():
                break
            evaluate(x)
        
        if not time_ok():
            return np.array(X_all)
        
        # Phase 2: Identify promising regions based on non-dominated grid points
        F_grid = np.array(F_all)
        
        # Find non-dominated points in the grid
        non_dominated_idx = []
        for i in range(len(F_grid)):
            dominated = False
            for j in range(len(F_grid)):
                if i == j:
                    continue
                if np.all(F_grid[j] <= F_grid[i]) and np.any(F_grid[j] < F_grid[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated_idx.append(i)
        
        # If no non-dominated points (shouldn't happen), use all
        if not non_dominated_idx:
            non_dominated_idx = list(range(len(F_grid)))
        
        # Phase 3: Local refinement using Nelder-Mead on promising directions
        # Use weighted scalarization to explore different trade-offs
        remaining_budget = budget - eval_count
        
        # Generate weight vectors for scalarization
        n_weights = min(20, remaining_budget // 10)
        if n_weights < 3:
            n_weights = 3
        
        # Simplex-lattice weight vectors
        weights = []
        for w1 in np.linspace(0.1, 0.8, 3):
            for w2 in np.linspace(0.1, 0.8 - w1, 2):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.1:
                    weights.append([w1, w2, w3])
        
        # If we need more weights, add random ones
        rng = np.random.default_rng(42)
        while len(weights) < n_weights:
            w = rng.random(n_obj)
            w /= w.sum()
            weights.append(w)
        
        weights = np.array(weights[:n_weights])
        
        # For each weight vector, run local refinement from the best grid point
        # in that direction
        for w in weights:
            if not time_ok():
                break
            
            # Find the grid point that minimizes the weighted sum
            weighted_sums = F_grid @ w
            best_grid_idx = np.argmin(weighted_sums)
            x0 = F_grid[best_grid_idx]  # This is wrong, should be X
            x0 = X_all[best_grid_idx].copy()
            
            # Nelder-Mead local search on the weighted scalarization
            def scalar_objective(x):
                if not time_ok():
                    return np.inf
                f = evaluate(x)
                return np.dot(f, w)
            
            # Simple Nelder-Mead implementation
            x_best = x0.copy()
            f_best = scalar_objective(x0)
            
            # Create initial simplex
            simplex = [x_best.copy()]
            f_simplex = [f_best]
            
            for i in range(dim):
                x_new = x_best.copy()
                x_new[i] += 0.1 * (hi[i] - lo[i])
                x_new = np.clip(x_new, lo, hi)
                simplex.append(x_new)
                f_simplex.append(scalar_objective(x_new))
            
            # Iterate Nelder-Mead
            max_iter = 50
            for it in range(max_iter):
                if not time_ok():
                    break
                
                # Sort by objective value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if abs(f_simplex[-1] - f_simplex[0]) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = centroid + (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, lo, hi)
                f_reflect = scalar_objective(x_reflect)
                
                if f_reflect < f_simplex[0]:
                    # Expansion
                    x_expand = centroid + 2 * (centroid - simplex[-1])
                    x_expand = np.clip(x_expand, lo, hi)
                    f_expand = scalar_objective(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[-2]:
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, lo, hi)
                    f_contract = scalar_objective(x_contract)
                    if f_contract < f_simplex[-1]:
                        simplex[-1] = x_contract
                        f_simplex[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = scalar_objective(simplex[i])
            
            # Update best
            if f_simplex[0] < f_best:
                x_best = simplex[0]
                f_best = f_simplex[0]
        
        # Phase 4: Final random sampling with local refinement on best points
        if time_ok():
            n_final = min(30, budget - eval_count)
            for _ in range(n_final):
                if not time_ok():
                    break
                x_rand = lo + (hi - lo) * rng.random(dim)
                f_rand = evaluate(x_rand)
                
                # Quick local refinement
                for _ in range(5):
                    if not time_ok():
                        break
                    x_perturb = x_rand + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    x_perturb = np.clip(x_perturb, lo, hi)
                    f_perturb = evaluate(x_perturb)
                    if np.all(f_perturb <= f_rand):
                        x_rand = x_perturb
                        f_rand = f_perturb
        
        return np.array(X_all)
# EVOLVE-BLOCK-END
