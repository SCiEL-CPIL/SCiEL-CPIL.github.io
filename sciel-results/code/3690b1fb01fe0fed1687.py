import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation:
        # 120 initial random points to ensure coverage
        # 80 for local search refinement (approx 20 points * 4 steps * 1.0 eval/step roughly, 
        # but we need to be careful with eval count in local search)
        
        # Let's do 100 initial points.
        # Then pick top 20 non-dominated and refine each with a limited local search.
        
        n_init = 100
        n_refine = 20
        evals_per_refine = 4  # 4 steps of local search, 1 eval per step? No, 1D search needs more.
        
        # Actually, 1D minimization on a smooth function needs ~5-10 evals to be precise.
        # With 4 dims, that's 20-40 evals per point.
        # 20 points * 20 evals = 400 evals. Too many.
        
        # Alternative: Use fewer points, or cheaper local search.
        # Let's use 50 initial points.
        # Pick top 10 non-dominated.
        # Refine each with 5 evals (1 per dim, moving towards better neighbor found by 2 evals? No).
        
        # Better: Use Nelder-Mead or similar? No, too expensive.
        # Let's stick to coordinate descent with a small number of steps.
        # For each of the 4 dims, try 2 perturbations (left, right). That's 8 evals per point.
        # 10 points * 8 evals = 80 evals.
        # Total: 50 + 80 = 130. Leaves 70 buffer.
        
        # Let's increase init to 60.
        # 60 init.
        # Top 10 non-dominated.
        # 10 * 8 = 80 refine.
        # Total 140.
        
        n_init = 60
        
        # 1. Initial Random Sampling
        X = lo + (hi - lo) * np.random.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # 2. Identify Non-Dominated Points
        def is_non_dominated(f, F):
            for other in F:
                if np.all(other <= f) and np.any(other < f):
                    return False
            return True
        
        # Get indices of non-dominated points
        nd_indices = [i for i in range(len(F)) if is_non_dominated(F[i], F)]
        
        # If not enough non-dominated, take best by hypervolume contribution or just random
        if len(nd_indices) < n_refine:
            # Fill with random points from the rest
            remaining = [i for i in range(len(F)) if i not in nd_indices]
            np.random.shuffle(remaining)
            nd_indices.extend(remaining[:n_refine - len(nd_indices)])
        
        # Select top n_refine
        refine_indices = nd_indices[:n_refine]
        
        # 3. Local Search Refinement
        # Strategy: For each selected point, perform coordinate-wise local search.
        # For each dimension, try a small step left and right.
        # Keep the best (non-dominated or better hypervolume contribution).
        
        # We need a fast way to decide if a new point is "better".
        # Since we want to maximize hypervolume, we prefer points that are non-dominated
        # and far from others. A simple proxy: keep if it's non-dominated w.r.t current pool,
        # or if it dominates an existing point.
        
        # To keep it simple and robust:
        # We will maintain a list of "refined" points.
        # For each starting point, we do a greedy hill climb on the objectives?
        # No, multi-objective.
        
        # Let's do a simple "neighbor" search:
        # For each dim j, try x[j] + delta and x[j] - delta.
        # If the new point is non-dominated w.r.t the current set of all evaluated points, accept it.
        
        all_X = list(X)
        all_F = list(F)
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def is_pareto_optimal(f, F_list):
            for other in F_list:
                if dominates(other, f):
                    return False
            return True

        # We will refine each of the n_refine points
        # We have budget = 200. Used = 60. Remaining = 140.
        # 140 / 10 points = 14 evals per point.
        # 4 dims. 3 evals per dim (current, left, right)? No, current is known.
        # 2 evals per dim (left, right). 8 evals per point.
        # Total refine evals = 10 * 8 = 80.
        # Total used = 60 + 80 = 140.
        # Safe.
        
        # Let's increase n_refine to 15 if budget allows?
        # 15 * 8 = 120. 60 + 120 = 180. Still safe.
        # Let's use n_refine = 15.
        
        n_refine = 15
        if len(refine_indices) < n_refine:
             remaining = [i for i in range(len(F)) if i not in refine_indices]
             np.random.shuffle(remaining)
             refine_indices.extend(remaining[:n_refine - len(refine_indices)])
        else:
             refine_indices = refine_indices[:n_refine]
             
        # Step size for local search
        # Use a small fraction of the range
        step_frac = 0.05
        
        for idx in refine_indices:
            if used >= self.budget or time.time() > deadline:
                break
                
            x = X[idx].copy()
            f = F[idx].copy()
            
            # For each dimension, try to improve
            for d in range(self.dim):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Try left
                x_left = x.copy()
                x_left[d] -= step_frac * (hi[d] - lo[d])
                x_left = np.clip(x_left, lo, hi)
                
                # Check if already evaluated? (Simple check: not strictly necessary if we just record)
                # But we must not exceed budget.
                
                f_left = problem(x_left)
                used += 1
                
                # Accept if it is non-dominated w.r.t the global set
                # To be fast, we can just check against the current best front or just add it.
                # Since we record ALL points, adding a dominated point doesn't hurt hypervolume
                # (it just doesn't contribute), but it wastes budget if it's bad.
                # However, "accepting" it as the new center for further search is risky if it's worse.
                
                # Better logic:
                # If f_left is non-dominated w.r.t all_F, it's a good candidate.
                # If it dominates f, it's definitely good.
                
                # Let's use a simple criterion:
                # If f_left is non-dominated w.r.t the initial population F (approx), keep it.
                # Actually, just check against the current global list of non-dominated points?
                # That's expensive.
                
                # Simpler: If f_left is better than f in at least one objective and not worse in others?
                # No, that's Pareto improvement.
                
                # Let's just collect all these neighbors.
                # We have 2 neighbors per dim * 4 dims = 8 neighbors.
                # We will add all of them to the global pool.
                # The hypervolume calculation will ignore the dominated ones.
                
                all_X.append(x_left)
                all_F.append(f_left)
                
                # Try right
                x_right = x.copy()
                x_right[d] += step_frac * (hi[d] - lo[d])
                x_right = np.clip(x_right, lo, hi)
                
                f_right = problem(x_right)
                used += 1
                
                all_X.append(x_right)
                all_F.append(f_right)
                
                # Optionally, if one of them is better, move x to that one for the next dim?
                # This turns it into a greedy local search.
                # Check if f_left or f_right dominates f.
                if dominates(f_left, f):
                    x = x_left
                    f = f_left
                elif dominates(f_right, f):
                    x = x_right
                    f = f_right
                # If neither dominates, keep original x for next dim?
                # Or pick the one that is "closer" to the ideal?
                # Let's stick to greedy Pareto improvement only.
                
        # Final cleanup: The evaluator records ALL points returned by problem().
        # My code calls problem() and appends to all_X/all_F, but the evaluator
        # records the return value of __call__?
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Wait, the contract says: "Every returned objective vector is recorded".
        # This usually implies the evaluator wraps the `problem` function to record calls.
        # "The evaluator ... calls np.random.seed(seed), creates Optimizer ... and calls it once."
        # "Every returned objective vector is recorded" -> This refers to the output of `problem(x)`.
        # So I don't need to return the points. The evaluator tracks them.
        
        return None
