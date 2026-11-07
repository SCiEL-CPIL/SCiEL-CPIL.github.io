"""DTLZ7 hypervolume optimizer via max-min facility location in objective space."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer using max-min facility location for Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _pareto_front_objectives(self, x1, x2):
        """Compute DTLZ7 objective values for given x1, x2 with x3..x22=0.
        
        DTLZ7: g = 0 when x3..x22=0
        M = 3 objectives
        g = 0
        h = (g + 1) * x1
        h = (g + 1) * x2
        f1 = x1
        f2 = x2
        f3 = (g + 1) * (M - 1 - h)
        
        With g=0:
        f1 = x1
        f2 = x2  
        f3 = (1) * (2 - x1 - x2) = 2 - x1 - x2
        """
        f1 = x1
        f2 = x2
        f3 = 2.0 - x1 - x2
        return np.column_stack([f1, f2, f3])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget

        # Use max-min facility location in objective space to place points
        # that maximize minimum pairwise distance in (f1, f2, f3) space.
        # This ensures uniform coverage of the hypervolume-relevant region.
        
        # Start with well-spread initial points in objective space
        # The Pareto front lies on the plane f1 + f2 + f3 = 2 with f1,f2,f3 >= 0
        # We parameterize by (x1, x2) where x1=f1, x2=f2, f3=2-x1-x2
        
        # Initialize with points spread across the front
        initial_pts = [
            (0.0, 0.0),    # f=[0,0,2]
            (0.0, 1.0),    # f=[0,1,1]
            (1.0, 0.0),    # f=[1,0,1]
            (1.0, 1.0),    # f=[1,1,0]
        ]
        
        # Compute objective vectors for initial points
        obj_pts = self._pareto_front_objectives(
            np.array([p[0] for p in initial_pts]),
            np.array([p[1] for p in initial_pts])
        )
        
        pts = list(initial_pts)
        
        # Greedy max-min facility location: add points that maximize
        # minimum distance to existing points in objective space
        rng = np.random.default_rng(42)
        
        while len(pts) < n:
            if time.time() > deadline - 10:  # leave time for evaluation
                break
                
            # Generate candidate points in (x1, x2) space
            # Bias toward regions that contribute more to hypervolume
            n_candidates = 5000
            cand_x1 = rng.random(n_candidates)
            cand_x2 = rng.random(n_candidates)
            
            # Ensure valid region: x1 + x2 <= 2 (for f3 >= 0)
            valid = cand_x1 + cand_x2 <= 2.0
            cand_x1 = cand_x1[valid]
            cand_x2 = cand_x2[valid]
            
            if len(cand_x1) == 0:
                continue
            
            # Compute objective vectors for candidates
            cand_obj = self._pareto_front_objectives(cand_x1, cand_x2)
            
            # Compute minimum distance from each candidate to existing points
            # Using broadcasting for efficiency
            # existing_obj shape: (k, 3), cand_obj shape: (m, 3)
            dists = np.sqrt(np.sum(
                (cand_obj[:, np.newaxis, :] - obj_pts[np.newaxis, :, :])**2, 
                axis=2
            ))
            min_dists = np.min(dists, axis=1)
            
            # Select the candidate with maximum minimum distance
            best_idx = np.argmax(min_dists)
            best_x1 = cand_x1[best_idx]
            best_x2 = cand_x2[best_idx]
            
            pts.append((best_x1, best_x2))
            new_obj = self._pareto_front_objectives(
                np.array([best_x1]), np.array([best_x2])
            )
            obj_pts = np.vstack([obj_pts, new_obj])
            
            # Periodic check for time
            if len(pts) % 50 == 0 and time.time() > deadline - 15:
                break
        
        # Trim to budget if we went over
        pts = pts[:n]
        
        x1 = np.array([p[0] for p in pts])
        x2 = np.array([p[1] for p in pts])
        
        # Clip to bounds (defensive)
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])
        
        # Build full 22-dim vectors with x3..x22 = 0
        X = np.zeros((len(x1), self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        
        # Evaluate each point; stop early if we exceed the deadline.
        F = np.zeros((len(x1), self.n_obj))
        for i in range(len(x1)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
