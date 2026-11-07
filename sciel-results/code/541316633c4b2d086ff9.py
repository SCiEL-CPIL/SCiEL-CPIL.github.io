import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (from task contract)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Population size: balance between exploration and exploitation
        # Using a moderate size to allow for sufficient generations and diversity
        pop_size = min(50, max(20, budget // 4))

        # Initialize population
        # Use a mix of random and center-biased points to cover the search space effectively
        rng = np.random.default_rng(0)
        X = np.empty((pop_size, dim))
        n_rand = pop_size // 2
        n_center = pop_size - n_rand

        X[:n_rand] = lo + span * rng.random((n_rand, dim))
        if n_center > 0:
            # Center-biased initialization
            center_samples = np.clip(0.5 + 0.2 * rng.standard_normal((n_center, dim)), 0, 1)
            X[n_rand:] = lo + span * center_samples

        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominated_mask(F_arr):
            """Return boolean mask where True means the point is dominated."""
            n = len(F_arr)
            mask = np.zeros(n, dtype=bool)
            for i in range(n):
                if mask[i]:
                    continue
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] and F[j] != F[i]
                dominates_i = np.all(F_arr <= F_arr[i], axis=1) & np.any(F_arr < F_arr[i], axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = True
            return mask

        def hypervolume_contribution(F_arr, ref):
            """Approximate hypervolume contribution for 3-objective case.
            Returns a scalar approximation of the hypervolume dominated by points in F_arr."""
            n = len(F_arr)
            if n == 0:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid_mask = np.all(F_arr < ref, axis=1)
            F_valid = F_arr[valid_mask]
            if len(F_valid) == 0:
                return 0.0
            
            # For 3-objective hypervolume, we can use a simple approximation
            # by sorting and computing the volume of boxes
            # This is an O(n^2) method but acceptable for small populations
            F_valid = np.array(F_valid)
            n_valid = len(F_valid)
            
            if n_valid == 1:
                return np.prod(ref - F_valid[0])
            
            # Sort by first objective
            idx = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[idx]
            
            hv = 0.0
            for i in range(n_valid):
                # For each point, calculate the volume it uniquely contributes
                # This is a simplification; true hypervolume is more complex
                # We approximate by assuming each point contributes a box
                # up to the next point in each dimension
                f_i = F_sorted[i]
                
                # Find the next point that is better in at least one dimension
                # This is a rough approximation
                box_size = np.zeros(n_obj)
                for j in range(n_obj):
                    # Find the minimum value in dimension j among points that are
                    # not dominated by f_i in all other dimensions
                    # Simplification: just use the reference point
                    box_size[j] = ref[j] - f_i[j]
                
                # This is a very rough approximation
                hv += np.prod(box_size)
            
            return hv

        while used < budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated parents
            nondom_mask = ~dominated_mask(F)
            if np.any(nondom_mask):
                parents_idx = np.where(nondom_mask)[0]
            else:
                parents_idx = np.arange(len(F))
            
            # Select a parent
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]
            
            # Mutate with adaptive step size
            # Start with larger steps, decrease over time
            progress = used / budget
            step_size = 0.5 * (1 - progress) + 0.05
            noise = rng.standard_normal(dim)
            child = parent + step_size * span * noise
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replacement strategy: replace the point that is dominated by the child if possible
            # Otherwise, replace the point with the largest sum (worst in terms of magnitude)
            
            # Check if child dominates any existing point
            child_dominates = np.all(F >= fc, axis=1) & np.any(F > fc, axis=1)
            if np.any(child_dominates):
                # Replace one of the dominated points (randomly)
                dominated_indices = np.where(child_dominates)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the point with the largest sum
                replace_idx = int(np.argmax(np.sum(F, axis=1)))
            
            X[replace_idx] = child
            F[replace_idx] = fc
            
            # Occasional restarts to escape local optima
            if used % 50 == 0 and used < budget - 20:
                # Replace a random point with a new random point
                restart_idx = rng.integers(len(X))
                X[restart_idx] = lo + span * rng.random(dim)
                F[restart_idx] = problem(X[restart_idx])
                used += 1

        return X
