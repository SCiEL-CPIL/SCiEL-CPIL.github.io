import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        eval_count = 0
        max_budget = self.budget

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= max_budget:
                return None
            eval_count += 1
            return np.array(problem(x))

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            if len(F) == 0:
                return False
            F = np.array(F)
            # f is dominated if exists j such that F[j] <= f and F[j] != f
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            F = np.array(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        def local_search(x_start, f_start, n_steps, w=None):
            """
            Perform local search around x_start.
            Returns the best non-dominated point found in the neighborhood.
            """
            x_best = x_start.copy()
            f_best = f_start.copy()
            
            # If a weight vector is provided, we can use it to guide the search
            # otherwise we just accept if non-dominated in the current set?
            # For simplicity, we'll just accept if it improves the weighted sum
            # or is non-dominated relative to x_best.
            
            for step in range(n_steps):
                if time.time() > deadline:
                    break
                # Random perturbation
                scale = 0.1 * (1.0 - step / float(n_steps))
                x_new = x_best + scale * np.random.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                
                # Acceptance criteria:
                # 1. If f_new is better than f_best in weighted sum
                # 2. Or if f_new is non-dominated by f_best and f_best is not dominated by f_new
                # For a simple local search on a single point, we just want to find a better point.
                # Let's use a simple rule: accept if f_new is better in at least one objective 
                # and not worse in any? That's too strict.
                # Let's use weighted sum with a random weight per step to explore different parts of the front.
                w_step = np.random.random(n_obj)
                w_step /= np.sum(w_step)
                
                if np.dot(f_new, w_step) < np.dot(f_best, w_step):
                    x_best = x_new
                    f_best = f_new
            return x_best, f_best

        # 1. Coarse Map: LHS Sampling
        n_initial = 40
        if n_initial > max_budget:
            n_initial = max_budget
            
        # Generate LHS samples
        samples = np.zeros((n_initial, dim))
        for d in range(dim):
            perm = np.random.permutation(n_initial)
            samples[:, d] = (perm + np.random.random(n_initial)) / n_initial
        samples = lo + (hi - lo) * samples
        
        # Evaluate initial samples
        X = []
        F = []
        for x in samples:
            if time.time() > deadline:
                break
            f = evaluate(x)
            if f is None:
                break
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)
        
        # 2. Identify Non-Dominated Points
        nd_indices = non_dominated_sort(F)
        
        # 3. Targeted Local Refinement
        # Refine each non-dominated point
        # Budget remaining
        remaining = max_budget - eval_count
        
        # Determine steps per point
        if len(nd_indices) > 0:
            steps_per_point = max(2, remaining // len(nd_indices))
        else:
            steps_per_point = 0
            
        # We will create a new list of refined points
        refined_X = []
        refined_F = []
        
        for idx in nd_indices:
            if time.time() > deadline:
                break
            if eval_count >= max_budget:
                break
                
            x_start = X[idx]
            f_start = F[idx]
            
            # Run local search
            x_ref, f_ref = local_search(x_start, f_start, steps_per_point)
            
            # Check if the refined point is non-dominated against the current refined set
            # To avoid O(N^2) checks, we just keep it for now and filter at the end
            # Or we can check against the original non-dominated set
            if not is_dominated(f_ref, F[nd_indices]):
                refined_X.append(x_ref)
                refined_F.append(f_ref)
            else:
                # If dominated, maybe keep the original?
                # Actually, if local search failed to improve, keep original
                refined_X.append(x_start)
                refined_F.append(f_start)
        
        # 4. If budget remains, do a few random restarts
        remaining = max_budget - eval_count
        n_restarts = min(10, remaining // 5)
        
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            if eval_count >= max_budget:
                break
            
            x_rand = lo + np.random.random(dim) * (hi - lo)
            f_rand = evaluate(x_rand)
            if f_rand is None:
                break
            
            # Local search on this random point
            x_ref, f_ref = local_search(x_rand, f_rand, 5)
            refined_X.append(x_ref)
            refined_F.append(f_ref)
            
        # Final cleanup: remove dominated points from the refined set
        if len(refined_F) > 0:
            refined_F = np.array(refined_F)
            refined_X = np.array(refined_X)
            final_nd = non_dominated_sort(refined_F)
            final_X = refined_X[final_nd]
            final_F = refined_F[final_nd]
        else:
            # Fallback
            final_X = np.array([lo + 0.5 * (hi - lo)])
            final_F = np.array([evaluate(final_X[0])])
            
        # The return value is ignored, but we must return something.
        # Returning the first non-dominated point or the entire array is fine.
        if len(final_X) > 0:
            return final_X[0]
        else:
            return X[0]

# EVOLVE-BLOCK-END
