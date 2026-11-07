import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initial Random Sampling to estimate ideal point z*
        n_init = 10
        X_init = lo + (hi - lo) * np.random.rand(n_init, self.dim)
        F_init = np.array([problem(x) for x in X_init])
        z_star = np.min(F_init, axis=0)
        
        used = n_init
        archive = list(zip(X_init, F_init))
        
        # 2. Define Weight Vectors for Tchebycheff Decomposition
        # For 3 objectives, use weights that cover the simplex
        # Weights must sum to 1 and be non-negative
        # Let's use a set of 6 weights: corners and edge midpoints
        weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5])
        ]
        
        n_weights = len(weights)
        budget_per_weight = (self.budget - used) // n_weights
        
        # 3. Local Search for each weight vector
        for i, w in enumerate(weights):
            if time.time() > deadline - 10: # Stop if close to deadline
                break
            
            # Starting point: use one of the initial random points or a random point
            # To ensure diversity, let's use the initial points cyclically
            start_x = X_init[i % n_init].copy()
            
            # Define the Tchebycheff scalarizing function
            def tchebycheff(x):
                if time.time() > deadline:
                    return 1e10
                fx = problem(x)
                # Tchebycheff function: max_i { w_i * |f_i(x) - z*_i| }
                # Since we are minimizing, and w_i > 0, we can use w_i * (f_i(x) - z*_i)
                # But to be safe with the definition, we use the absolute difference
                # However, for minimization, f_i(x) >= z*_i usually holds near the front
                # Standard Tchebycheff: max_i { w_i * |f_i(x) - z*_i| }
                diffs = np.abs(fx - z_star)
                return np.max(w * diffs)
            
            # Nelder-Mead Local Search
            # We need to count function calls
            eval_count = [0]
            max_evals = budget_per_weight
            
            def count_problem(x):
                if eval_count[0] >= max_evals or time.time() > deadline:
                    return 1e10
                eval_count[0] += 1
                used_evals[0] += 1
                fx = problem(x)
                # Record in archive
                archive.append((x, fx))
                return fx
            
            # We need to modify the tchebycheff function to use count_problem
            # But Nelder-Mead calls the objective function directly.
            # So we wrap the objective function.
            
            def objective(x):
                if eval_count[0] >= max_evals or time.time() > deadline:
                    return 1e10
                eval_count[0] += 1
                used_evals[0] += 1
                fx = problem(x)
                archive.append((x, fx))
                # Calculate Tchebycheff value
                diffs = np.abs(fx - z_star)
                return np.max(w * diffs)
            
            # Simple Nelder-Mead Implementation
            # Initial simplex
            simplex = [start_x]
            for j in range(self.dim):
                x_j = start_x.copy()
                step = 0.1 * (hi[j] - lo[j]) if (hi[j] - lo[j]) > 0 else 0.1
                x_j[j] += step
                simplex.append(x_j)
            
            simplex_f = [objective(x) for x in simplex]
            
            # Nelder-Mead Iterations
            while eval_count[0] < max_evals and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = objective(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    f_exp = objective(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = objective(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for j in range(1, self.dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            simplex_f[j] = objective(simplex[j])
                
                # Check for convergence
                if np.std(simplex_f) < 1e-6:
                    break

        # 4. Return the best points (though return value is ignored, we return X for consistency)
        # We can return the X part of the archive
        return [x for x, f in archive]

# EVOLVE-BLOCK-END
