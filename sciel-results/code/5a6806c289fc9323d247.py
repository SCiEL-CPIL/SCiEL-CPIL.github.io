import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        rng = np.random.default_rng(0)
        
        # Track non-dominated points found so far
        # front_X: list of vectors
        # front_F: list of objective vectors
        front_X = []
        front_F = []
        
        used = 0
        budget = self.budget
        dim = self.dim
        
        # Helper to check if a point is dominated by any in front
        def is_dominated(f_new, front_F):
            for f in front_F:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
            
        # Helper to remove dominated points from front
        def update_front(f_new, x_new):
            # Check if new point is dominated
            if is_dominated(f_new, front_F):
                return
            
            # Add new point
            front_X.append(x_new)
            front_F.append(f_new)
            
            # Remove points dominated by new point
            # Note: modifying list while iterating is bad, so create new list
            new_front_X = []
            new_front_F = []
            for i, f in enumerate(front_F):
                if i == len(front_F) - 1: # Just added
                    continue
                # Check if f is dominated by f_new
                if np.all(f_new <= f) and np.any(f_new < f):
                    continue # f is dominated by f_new, remove it
                new_front_X.append(front_X[i])
                new_front_F.append(f)
            
            # Reassign
            # We need to be careful with the just added point
            # Actually, simpler:
            # Keep the new point and any old points not dominated by it.
            # The loop above skips the last one (new point) from the "old" check, but we need to include the new point in the result.
            # Let's rewrite update_front more clearly.
            
        def clean_front():
            # Remove any points in front_F that are dominated by another point in front_F
            # This is O(N^2) but N is small (front size)
            keep_X = []
            keep_F = []
            n = len(front_F)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    if np.all(front_F[j] <= front_F[i]) and np.any(front_F[j] < front_F[i]):
                        dominated = True
                        break
                if not dominated:
                    keep_X.append(front_X[i])
                    keep_F.append(front_F[i])
            front_X[:] = keep_X
            front_F[:] = keep_F

        # 1. Initialization
        init_size = 20
        for _ in range(init_size):
            x = lo + span * rng.random(dim)
            f = problem(x)
            used += 1
            front_X.append(x)
            front_F.append(f)
        
        # Clean the initial front
        clean_front()
        
        # 2. Evolution
        while used < budget:
            # Select parent
            if len(front_X) == 0:
                # Fallback: random
                parent = lo + span * rng.random(dim)
            else:
                # Pick random from front
                idx = rng.integers(len(front_X))
                parent = front_X[idx]
            
            # Mutate
            # Adaptive step? Let's use a fixed step of 0.1 * span
            # Or maybe 0.2? WFG9 is multimodal.
            step = 0.1 * span
            mutation = rng.standard_normal(dim) * step
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update front
            # Check if child is dominated by current front
            is_dom = False
            for f in front_F:
                if np.all(f <= fc) and np.any(f < fc):
                    is_dom = True
                    break
            
            if not is_dom:
                front_X.append(child)
                front_F.append(fc)
                # Clean front (remove points dominated by the new one or others)
                clean_front()
            else:
                # Even if dominated, we might want to keep it? 
                # No, for selection, only non-dominated are useful parents.
                # But since all evaluations are recorded, the dominated one is still counted in HV.
                # We just don't use it for selection.
                pass

        return None
