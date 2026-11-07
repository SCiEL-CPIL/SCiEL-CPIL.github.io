"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Global Random Sampling and Local Scalarized Refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Global Initialization
        # Use 50% of the budget for global exploration
        init_size = self.budget // 2
        if init_size < 10:
            init_size = 10
            
        X_init = np.random.rand(init_size, self.dim)
        X_init = lo + (hi - lo) * X_init
        
        F_init = np.array([problem(x) for x in X_init])
        used = init_size
        
        # 2. Identify Non-Dominated Points
        N = len(F_init)
        is_dominated = np.zeros(N, dtype=bool)
        for i in range(N):
            if is_dominated[i]:
                continue
            # Check if any j dominates i
            # j dominates i if F[j] <= F[i] and F[j] < F[i]
            dominates_i = np.all(F_init <= F_init[i], axis=1) & np.any(F_init < F_init[i], axis=1)
            if np.any(dominates_i):
                is_dominated[i] = True
                
        non_dom_idx = np.where(~is_dominated)[0]
        
        # 3. Select Candidates for Refinement
        # We want to refine points that are likely to contribute to the hypervolume.
        # A simple heuristic: pick the non-dominated points that are most "extreme" or have best scalarization.
        # Let's pick up to 10 candidates.
        n_refine = min(10, len(non_dom_idx))
        
        # If we have few non-dominated points, pick all. Otherwise, pick diverse ones.
        # For simplicity, let's pick the ones with the lowest sum of normalized objectives?
        # Or just random non-dominated points?
        # Better: Pick points that maximize the minimum distance to other non-dominated points (greedy diversity).
        
        if n_refine == 0:
             # Fallback if all dominated (unlikely with random init)
             candidates_idx = np.random.choice(N, min(10, N), replace=False)
        else:
             # Greedy diversity selection from non-dominated set
             candidates_idx = []
             remaining = list(non_dom_idx)
             # Start with a random point
             first_idx = remaining.pop(np.random.randint(len(remaining)))
             candidates_idx.append(first_idx)
             
             for _ in range(n_refine - 1):
                 if not remaining:
                     break
                 # For each candidate in remaining, calculate min distance to already selected
                 # Use Euclidean distance in objective space
                 selected_F = F_init[candidates_idx]
                 best_candidate = -1
                 max_min_dist = -1
                 
                 # This is O(n_refine * n_non_dom * n_selected), which is small
                 for r in remaining:
                     dists = np.linalg.norm(selected_F - F_init[r], axis=1)
                     min_dist = np.min(dists)
                     if min_dist > max_min_dist:
                         max_min_dist = min_dist
                         best_candidate = r
                 
                 if best_candidate != -1:
                     candidates_idx.append(best_candidate)
                     remaining.remove(best_candidate)
                 
        # 4. Local Refinement
        # Distribute remaining budget among candidates
        remaining_budget = self.budget - used
        if remaining_budget <= 0:
            return X_init
            
        budget_per_candidate = max(1, remaining_budget // len(candidates_idx))
        
        # Generate weight vectors for scalarization
        # Use random weights on the simplex
        weights = np.random.dirichlet(np.ones(self.n_obj), size=len(candidates_idx))
        
        for i, idx in enumerate(candidates_idx):
            if used >= self.budget:
                break
                
            x0 = X_init[idx].copy()
            w = weights[i]
            
            def scalarized_obj(x):
                x_clipped = np.clip(x, lo, hi)
                f_vals = problem(x_clipped)
                return np.dot(w, f_vals)
            
            # Use Nelder-Mead for local optimization
            # maxiter is roughly the number of function evaluations
            # We allow a few extra to be safe, but we must respect the global budget
            # Nelder-Mead uses ~dim+1 initial points, then iterates.
            # To strictly control budget, we can't easily hook into Nelder-Mead's eval count 
            # without wrapping. 
            # However, with dim=4, 10-20 iterations is ~30-60 evals.
            # We have ~100 budget left for 10 candidates -> 10 evals each.
            # This might be too tight for Nelder-Mead to converge well.
            
            # Alternative: Coordinate Ascent or Simple Random Walk?
            # Let's use a simple local search: 
            # For k steps, try moving in random directions and keep if better in scalarized obj.
            
            current_x = x0.copy()
            current_f = F_init[idx].copy()
            current_scalar = np.dot(w, current_f)
            
            steps = budget_per_candidate
            step_size = 0.1 * (hi - lo)
            
            for _ in range(steps):
                if used >= self.budget:
                    break
                # Generate a random direction
                direction = np.random.randn(self.dim)
                direction /= np.linalg.norm(direction)
                
                new_x = current_x + step_size * direction
                new_x = np.clip(new_x, lo, hi)
                
                new_f = problem(new_x)
                used += 1
                
                new_scalar = np.dot(w, new_f)
                
                if new_scalar < current_scalar:
                    current_x = new_x
                    current_f = new_f
                    current_scalar = new_scalar
                    # Optionally reduce step size
                    step_size *= 0.9
                else:
                    # Try a smaller step in the same direction? Or just skip.
                    # Skipping is safer for budget.
                    pass

        return X_init
# EVOLVE-BLOCK-END
