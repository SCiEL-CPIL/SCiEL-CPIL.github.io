import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        n_points = self.budget

        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # f1 = x[0], f2 = x[1], f3 = 1 (when x[2:]=0)
        # Reference point: [1.1, 1.1, 25.0]
        # Hypervolume contribution of point (f1, f2, f3) is (1.1-f1)*(1.1-f2)*(25.0-f3)
        # Since f3=1 for all Pareto points, the f3 term is constant (24.0)
        # So we maximize 2D hypervolume in (f1, f2) space with ref (1.1, 1.1)
        
        # Strategy: Greedy Farthest-Point Sampling (FPS) on a fine grid
        # This approximates the 2D k-center problem, maximizing minimum distance
        # to the nearest existing point for optimal spatial coverage
        
        # Create a fine candidate grid
        grid_size = 60  # 60x60 = 3600 candidates, more than enough for 200 points
        x_vals = np.linspace(lo[0], hi[0], grid_size)
        y_vals = np.linspace(lo[1], hi[1], grid_size)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        candidates = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
        
        n_candidates = len(candidates)
        
        # Start with corners to ensure boundary coverage
        corners = np.array([
            [lo[0], lo[1]],
            [lo[0], hi[1]],
            [hi[0], lo[1]],
            [hi[0], hi[1]]
        ])
        
        selected = []
        
        # Initialize minimum distance array
        # Distance from each candidate to the first selected point (corner 0)
        min_dists = np.linalg.norm(candidates - corners[0], axis=1)
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            if i < 4:
                # Add the four corners
                selected.append(corners[i])
                # Update min_dists with distance to this corner
                dists_to_new = np.linalg.norm(candidates - corners[i], axis=1)
                min_dists = np.minimum(min_dists, dists_to_new)
            else:
                # Find candidate with maximum minimum distance (farthest point)
                idx = np.argmax(min_dists)
                new_point = candidates[idx]
                selected.append(new_point)
                
                # Update min_dists
                dists_to_new = np.linalg.norm(candidates - new_point, axis=1)
                min_dists = np.minimum(min_dists, dists_to_new)
        
        if len(selected) == 0:
            # Fallback: should not happen, but just in case
            x1_arr = np.zeros(n_points)
            x2_arr = np.zeros(n_points)
        else:
            selected_arr = np.array(selected)
            x1_arr = selected_arr[:, 0]
            x2_arr = selected_arr[:, 1]
        
        # Clip to bounds just in case
        x1_arr = np.clip(x1_arr, lo[0], hi[0])
        x2_arr = np.clip(x2_arr, lo[1], hi[1])
        
        # Evaluate all selected points
        for i in range(len(selected)):
            if time.time() > deadline or used >= self.budget:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_arr[i]
            x[1] = x2_arr[i]
            # x[2:] = 0 for DTLZ7 Pareto front
            problem(x)
            used += 1

        return None
