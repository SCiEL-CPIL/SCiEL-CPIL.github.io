"""Multi-objective optimizer on RE37 using space-filling initialization and farthest-point sampling in decision space."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer that:
    1. Generates a large pool of candidates using a Halton sequence (space-filling).
    2. Iteratively selects the next point to evaluate by maximizing the minimum Euclidean distance
       to all previously evaluated points in the DECISION space.
    3. This ensures uniform coverage of the decision space, which heuristically leads to
       uniform coverage of the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim, skip=1):
        """Generate n points in [0,1)^dim using the Halton sequence, skipping the first `skip` points."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for base in range(2, dim + 2):  # primes: 2, 3, 5, 7, ...
                f = 1.0
                r = 0.0
                n_val = i
                while n_val > 0:
                    f /= base
                    r += f * (n_val % base)
                    n_val //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate a large pool of candidates using Halton sequence
        # We use a pool larger than the budget to allow for FPS selection
        pool_size = self.budget * 10
        halton_unit = self._halton_sequence(pool_size, self.dim, skip=1)
        X_pool = lo + (hi - lo) * halton_unit
        
        # Initialize tracking
        selected_X = np.empty((0, self.dim))
        evaluated_F = []
        
        # Indices of unevaluated points in the pool
        unevaluated_indices = list(range(pool_size))
        
        for _ in range(self.budget):
            if len(unevaluated_indices) == 0:
                break
            
            # Get the current pool of unevaluated candidates
            X_candidates = X_pool[unevaluated_indices]
            
            if len(selected_X) == 0:
                # First point: pick the center or just the first one
                # Center is usually a good start
                # Or just index 0
                best_idx_in_candidates = 0
            else:
                # Compute minimum distance from each candidate to the selected set
                # X_candidates: (M, D)
                # selected_X: (K, D)
                
                # Efficient computation
                # diff shape: (M, K, D)
                diff = X_candidates[:, np.newaxis, :] - selected_X[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dists = np.min(dists, axis=1)
                
                # Pick the candidate with the maximum minimum distance
                best_idx_in_candidates = np.argmax(min_dists)
            
            # Retrieve the best point
            best_idx_global = unevaluated_indices[best_idx_in_candidates]
            x_best = X_pool[best_idx_global]
            
            # Remove from unevaluated
            unevaluated_indices.pop(best_idx_in_candidates)
            
            # Add to selected
            selected_X = np.vstack([selected_X, x_best])
            
            # Evaluate
            f_best = problem(x_best)
            evaluated_F.append(f_best)

        # The return value is ignored, but we must return something
        # We return the last evaluated point or an array of them?
        # The contract says "return value of __call__ is ignored".
        return np.array(evaluated_F)
# EVOLVE-BLOCK-END
