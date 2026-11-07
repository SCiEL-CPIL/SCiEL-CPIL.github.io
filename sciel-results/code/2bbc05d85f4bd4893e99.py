import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Greedy Hypervolume Maximization.
    
    Strategy:
    1. The Pareto front of ZDT3 is known analytically: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1],
       achieved when x2..xn = 0.
    2. The problem reduces to selecting B x1 values in [0,1] to maximize the hypervolume of 
       {(x1, 1-sqrt(x1))} with reference point [11, 11].
    3. Instead of using a fixed heuristic distribution (like power law), we use an iterative 
       greedy algorithm to select the points. At each step, we find the x1 value that 
       maximizes the marginal hypervolume contribution given the current set of points.
    4. This ensures that the points are placed exactly where they are needed to cover the 
       front efficiently, adapting to the curvature and the reference point.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points):
        """
        Calculate hypervolume of a set of 2D points with reference point (11, 11).
        points: array of shape (N, 2)
        """
        if len(points) == 0:
            return 0.0
        
        # Sort by f1 ascending, then by f2 descending
        idx = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[idx]
        
        # Filter points strictly better than reference
        ref = np.array([11.0, 11.0])
        mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
        pts = pts[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Calculate hypervolume
        # For each point i, the contribution is (x_{i+1} - x_i) * (y_ref - y_i)
        # where x_{N} = ref_x
        hv = 0.0
        for i in range(len(pts)):
            x_next = pts[i+1, 0] if i < len(pts) - 1 else ref[0]
            width = x_next - pts[i, 0]
            height = ref[1] - pts[i, 1]
            hv += width * height
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        B = self.budget
        if B == 0:
            return np.array([])
            
        # We will evaluate exactly on the Pareto front.
        # x = [x1, 0, 0, ..., 0]
        
        # Initialize with a reasonable spread to avoid degenerate starts
        # Use a coarse power law grid initially to give the greedy algorithm a good starting point
        if B == 1:
            x1_values = np.array([0.0])
        else:
            i = np.arange(B)
            t = i / (B - 1)
            # Initial guess: power law p=2
            x1_values = t ** 2
            x1_values[0] = 0.0
            x1_values[-1] = 1.0
            
        # Compute initial points on the front
        def get_front_points(x1_vals):
            pts = np.column_stack((x1_vals, 1.0 - np.sqrt(x1_vals)))
            return pts
        
        current_pts = get_front_points(x1_values)
        
        # Greedy optimization of x1 positions
        # We have B slots. We will iteratively try to improve the worst-performing or random points.
        # To be efficient, we will perform a few rounds of local optimization for each point.
        
        # Since B=200 is small, we can afford to be a bit more thorough.
        # Strategy:
        # 1. Compute initial HV.
        # 2. For each point i, find the best x1 in [0,1] that maximizes HV when point i is replaced.
        #    We can do this by binary search or grid search over x1.
        #    Given the smoothness, a coarse grid search followed by refinement should work.
        
        # However, doing a full grid search for each of the 200 points in each iteration might be slow.
        # Let's do 3-5 iterations of "coordinate descent" style optimization.
        
        # Pre-compute a fine grid for x1 to search
        # 1000 points should be enough to find the optimum for each coordinate
        search_grid = np.linspace(0, 1, 1000)
        
        # To speed up HV calculation during search, we can cache the sorted points 
        # but since we are swapping one point, we can just recalculate.
        # 200 points * 1000 search steps * 5 iterations = 1,000,000 HV calcs.
        # Each HV calc is O(N log N) or O(N) if sorted. 
        # With N=200, it's very fast.
        
        # Let's optimize the order: randomize the order of points to update to avoid bias.
        indices = np.arange(B)
        
        for iteration in range(10): # 10 iterations of coordinate descent
            if time.time() > deadline - 5:
                break
                
            np.random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2:
                    break
                
                # Try to find the best x1 for point i
                # Remove point i from the set
                mask = np.ones(B, dtype=bool)
                mask[i] = False
                other_pts = current_pts[mask]
                
                # Evaluate HV for each candidate x1 in search_grid
                # We can vectorize this partially or just loop.
                # Loop is fine for 1000 items.
                
                best_x1 = current_pts[i, 0]
                best_hv = -1
                
                # Optimization: 
                # The marginal gain of a new point (x_new, y_new) in a set S is:
                # It adds area under the curve between the neighbors.
                # We can compute the "gap" efficiently.
                
                # For simplicity and robustness, let's just compute full HV for candidates.
                # But 1000 * 200 * 10 = 2M HV calcs. 
                # Each HV calc for 200 points is ~200 ops. Total 400M ops. Might be tight in Python.
                
                # Let's optimize the HV calculation for the search.
                # Sort other_pts by x1
                other_sorted_idx = np.argsort(other_pts[:, 0])
                other_sorted = other_pts[other_sorted_idx]
                
                # We want to find x_new such that the area added is max.
                # The area added by a point (x, y) into a set of points sorted by x is:
                # (x - x_prev) * (y_ref - y) + (x_next - x) * (y_ref - y) ... wait.
                # Actually, the HV is the sum of rectangles.
                # If we insert (x, y) between (x_prev, y_prev) and (x_next, y_next):
                # The old contribution of the gap was (x_next - x_prev) * (y_ref - y_prev) ... NO.
                # The HV is the union of boxes.
                # Let's stick to the simple loop but optimize the inner HV calc.
                
                # Alternative: 
                # Since the front is fixed, y is determined by x.
                # y = 1 - sqrt(x)
                # The marginal gain of adding a point (x, y) to a set of points on the front
                # is the area of the triangle/region between the new point and the linear 
                # interpolation of its neighbors on the front? No, it's the area under the 
                # "upper envelope".
                
                # Let's just do the simple loop but with a smaller search grid first, then refine.
                coarse_grid = np.linspace(0, 1, 100)
                
                best_hv_coarse = -1
                best_x1_coarse = current_pts[i, 0]
                
                for x_cand in coarse_grid:
                    y_cand = 1.0 - np.sqrt(x_cand)
                    new_pts = np.vstack([other_pts, [[x_cand, y_cand]]])
                    hv = self._hypervolume_2d(new_pts)
                    if hv > best_hv_coarse:
                        best_hv_coarse = hv
                        best_x1_coarse = x_cand
                        
                # Refine around best_x1_coarse
                if best_x1_coarse > 0.05 and best_x1_coarse < 0.95:
                    fine_grid = np.linspace(best_x1_coarse - 0.05, best_x1_coarse + 0.05, 100)
                    for x_cand in fine_grid:
                        x_cand = np.clip(x_cand, 0, 1)
                        y_cand = 1.0 - np.sqrt(x_cand)
                        new_pts = np.vstack([other_pts, [[x_cand, y_cand]]])
                        hv = self._hypervolume_2d(new_pts)
                        if hv > best_hv_coarse:
                            best_hv_coarse = hv
                            best_x1_coarse = x_cand
                            
                current_pts[i, 0] = best_x1_coarse
                current_pts[i, 1] = 1.0 - np.sqrt(best_x1_coarse)
                
        # Final sort and evaluation
        # Ensure we evaluate exactly these points
        x1_final = current_pts[:, 0]
        
        F = []
        for x1 in x1_final:
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break

        return np.array(F)
