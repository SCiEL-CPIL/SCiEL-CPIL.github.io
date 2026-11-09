"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using Weighted Sum Method with local refinement.
    
    Strategy:
    1. Generate a diverse set of weight vectors covering the objective space.
    2. For each weight vector, solve the single-objective weighted sum problem
       using a local search (Nelder-Mead) initialized from random points.
    3. Collect all non-dominated solutions found.
    4. Use remaining budget for a final local refinement pass on the best candidates.
    
    This exploits WFG7's known Pareto front structure by directly targeting
    different regions of the front through weight vector decomposition.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Track all evaluated solutions
        all_X = []
        all_F = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            eval_count += 1
            return x_clipped, f
        
        def is_pareto_optimal(F):
            """Check which points in F are non-dominated."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # If j dominates i, mark i as dominated
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated
        
        # Generate weight vectors covering the 3-objective simplex
        # Use a grid of weights that sum to 1
        n_weights = 20
        weights = []
        for w1 in np.linspace(0.1, 0.8, n_weights):
            for w2 in np.linspace(0.1, 0.8 - w1, max(1, int((0.8 - w1) * 2))):
                w3 = 1.0 - w1 - w2
                if w3 > 0.05:
                    weights.append(np.array([w1, w2, w3]))
        
        # Limit to fit budget
        max_weights = min(len(weights), 40)
        selected_weights = weights[:max_weights]
        
        # For each weight vector, do a local search
        results = []
        
        for w in selected_weights:
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Try 2-3 random starts for this weight vector
            n_starts = 2
            for start_idx in range(n_starts):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Random initialization
                x0 = lo + (hi - lo) * rng.random(self.dim)
                x0, f0 = evaluate(x0)
                
                # Local search using coordinate descent / Nelder-Mead style
                best_x = x0.copy()
                best_f = f0.copy()
                best_wsum = np.dot(w, best_f)
                
                # Simple local search: try small perturbations
                step_size = 0.05
                for iteration in range(15):
                    if eval_count >= self.budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    # Try perturbing each dimension
                    improved = False
                    for dim_idx in range(self.dim):
                        if eval_count >= self.budget:
                            break
                        
                        # Try increasing
                        x_try = best_x.copy()
                        x_try[dim_idx] += step_size
                        x_try, f_try = evaluate(x_try)
                        wsum_try = np.dot(w, f_try)
                        
                        if wsum_try < best_wsum:
                            best_x = x_try.copy()
                            best_f = f_try.copy()
                            best_wsum = wsum_try
                            improved = True
                        
                        # Try decreasing
                        x_try = best_x.copy()
                        x_try[dim_idx] -= step_size
                        x_try, f_try = evaluate(x_try)
                        wsum_try = np.dot(w, f_try)
                        
                        if wsum_try < best_wsum:
                            best_x = x_try.copy()
                            best_f = f_try.copy()
                            best_wsum = wsum_try
                            improved = True
                    
                    if not improved:
                        step_size *= 0.5
                        if step_size < 0.001:
                            break
                    else:
                        step_size = max(step_size * 0.7, 0.005)
                
                # Record the best solution for this weight vector
                results.append((best_x, best_f))
        
        # Collect all non-dominated solutions
        if len(results) > 0:
            F_arr = np.array([r[1] for r in results])
            X_arr = np.array([r[0] for r in results])
            
            # Filter to only points better than reference point [3.0, 5.0, 7.0]
            ref = np.array([3.0, 5.0, 7.0])
            mask = np.all(F_arr < ref, axis=1)
            F_arr = F_arr[mask]
            X_arr = X_arr[mask]
            
            # Find non-dominated
            if len(F_arr) > 0:
                pareto_mask = is_pareto_optimal(F_arr)
                final_X = X_arr[pareto_mask]
                final_F = F_arr[pareto_mask]
            else:
                final_X = X_arr
                final_F = F_arr
        else:
            final_X = np.empty((0, self.dim))
            final_F = np.empty((0, self.n_obj))
        
        # If we have budget left, do a final refinement pass
        remaining_budget = self.budget - eval_count
        if remaining_budget > 10 and len(final_F) > 0 and time.time() < deadline - 5:
            # Refine the best hypervolume contributors
            # Simple approach: try small perturbations around each Pareto point
            for i in range(len(final_F)):
                if remaining_budget <= 0:
                    break
                if time.time() > deadline - 3:
                    break
                
                x_base = final_X[i]
                f_base = final_F[i]
                
                # Try a few random perturbations
                for perturb in range(3):
                    if remaining_budget <= 0:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    x_try = x_base + 0.02 * rng.standard_normal(self.dim)
                    x_try, f_try = evaluate(x_try)
                    remaining_budget -= 1
                    
                    # Check if this new point is non-dominated with respect to current set
                    # and better than base
                    if np.all(f_try < ref):
                        # Check if it dominates any existing or is non-dominated
                        is_new_pareto = True
                        for j in range(len(final_F)):
                            if np.all(final_F[j] <= f_try) and np.any(final_F[j] < f_try):
                                is_new_pareto = False
                                break
                        
                        if is_new_pareto:
                            # Check if it dominates any existing points
                            new_final_F = [f_try]
                            new_final_X = [x_try]
                            for j in range(len(final_F)):
                                if not (np.all(f_try <= final_F[j]) and np.any(f_try < final_F[j])):
                                    new_final_F.append(final_F[j])
                                    new_final_X.append(final_X[j])
                            
                            final_F = np.array(new_final_F)
                            final_X = np.array(new_final_X)
        
        return final_X
# EVOLVE-BLOCK-END
