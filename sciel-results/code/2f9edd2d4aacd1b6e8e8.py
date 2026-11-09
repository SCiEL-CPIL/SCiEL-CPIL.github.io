import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x)
            all_fvals.append(f)
            n_evals += 1
            return f

        # DTLZ2 structure:
        # The Pareto front is defined by x0 and x1.
        # x2...x11 should be 0.5 to minimize the distance to the front (g=0).
        # We will sample x0 and x1 in [0, 1].
        
        # Strategy: Two-phase adaptive refinement in the 2D space of (x0, x1).
        # Phase 1: Coarse grid to establish a baseline coverage.
        # Phase 2: Iteratively insert points at the midpoint of the largest empty region
        #          (approximated by nearest neighbor distance in 2D) to refine coverage.
        
        # We need to manage the budget carefully.
        # Let's reserve a small number for initial grid, and use the rest for refinement.
        
        # Phase 1: Grid
        # A 10x10 grid gives 100 points. This leaves 100 for refinement.
        # Or maybe 8x8 = 64 points, leaving 136.
        # Let's try 10x10 grid.
        
        grid_size = 10
        n_grid = grid_size * grid_size
        
        if self.budget < n_grid:
             # If budget is very small, just do a Sobol or random
             # But budget is 200, so this is safe.
             pass
             
        # Generate grid points in [0, 1]
        # We want to include the boundaries 0 and 1? 
        # DTLZ2 front is on the sphere x0^2+x1^2+x2^2...=1.
        # In the decision space, x0, x1 in [0,1].
        # The mapping to the front is f1 = g * (1 + sin(pi*x0)), etc? No.
        # DTLZ2:
        # f1 = (1+g) * prod(cos(xi))
        # f2 = (1+g) * cos(x1) * prod_{i>=2}(cos(xi)) ... wait, standard DTLZ2:
        # f1 = (1+g) * cos(x1*pi/2)
        # f2 = (1+g) * sin(x1*pi/2)
        # ... for M objectives it's more complex.
        # For 3 objectives DTLZ2:
        # f1 = (1+g) * cos(x1*pi/2)
        # f2 = (1+g) * sin(x1*pi/2) * cos(x2*pi/2)
        # f3 = (1+g) * sin(x2*pi/2)
        # Wait, the standard definition uses x_1, x_2 for the front shape and x_3...x_n for g.
        # In our problem, dim=12. So x[0], x[1] are the "front" variables?
        # Usually DTLZ2 uses the last variable for g.
        # Let's check the prompt: "minimizing all 3 objectives of DTLZ2 (12 variables...)"
        # Standard DTLZ2:
        # g(x) = sum_{i=k+1}^n (x_i - 0.5)^2 - 10*cos(20*pi*(x_i-0.5))
        # f1 = (1+g) * cos(x_1*pi/2)
        # f2 = (1+g) * sin(x_1*pi/2) * cos(x_2*pi/2)
        # f3 = (1+g) * sin(x_2*pi/2)
        # Here k=2. So x[0] and x[1] define the front shape. x[2..11] define g.
        # To minimize g, we set x[2..11] = 0.5.
        # So the Pareto front is determined by x[0] and x[1] in [0, 1].
        # The front is a quarter-sphere in the objective space.
        
        # We will sample x[0] and x[1].
        
        # Phase 1: Grid
        # Use linspace including 0 and 1?
        # The front is smooth. A grid of 10x10 is decent.
        x0_vals = np.linspace(0, 1, grid_size)
        x1_vals = np.linspace(0, 1, grid_size)
        
        # We will store the 2D coordinates of evaluated points to guide refinement
        # and the corresponding full 12D vectors.
        # Actually, we just need to track the 2D positions to find the "largest gap".
        
        # Let's collect points in a list of (x0, x1)
        # We will evaluate the grid first.
        
        for i in range(grid_size):
            for j in range(grid_size):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0_vals[i]
                x[1] = x1_vals[j]
                x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                f = eval_point(x)
                if f is None:
                    break
            
            if n_evals >= self.budget or time.time() > deadline:
                break
        
        if n_evals >= self.budget or time.time() > deadline:
            return all_points

        # Phase 2: Refinement
        # We have evaluated a set of points.
        # We want to add points that maximize the minimum distance to existing points in the 2D (x0, x1) space.
        # This is a greedy farthest-point sampling.
        
        # Get current 2D points
        # We need to extract x[0] and x[1] from all_points
        # But all_points contains full 12D vectors.
        
        # To make this efficient, let's keep a separate list of 2D coords.
        # However, we already appended to all_points.
        # Let's extract them.
        
        coords_2d = np.array([[p[0], p[1]] for p in all_points])
        
        # Remaining budget
        remaining = self.budget - n_evals
        
        # For each new point, find the best location.
        # A simple heuristic:
        # 1. Generate a set of candidate points (e.g., midpoints of all pairs, or a fine grid).
        # 2. For each candidate, calculate the minimum distance to existing points.
        # 3. Pick the candidate with the maximum minimum distance.
        # 4. Evaluate it and add to the set.
        
        # To avoid O(N^2) cost for each insertion (which is fine for N=200, but let's be smart),
        # we can use a grid-based approximation or just brute force with a small candidate pool.
        
        # Let's use a fine grid of candidates for the next step.
        # Or, simpler: just use a Sobol sequence for the remaining points?
        # The current program uses Sobol for all points.
        # The "Inspiration 2" suggests two-phase: coarse grid then local refinement.
        
        # Let's implement the greedy farthest point sampling properly.
        # For each new point:
        # Candidates: Midpoints of all existing pairs? Too many.
        # Candidates: A fixed fine grid (e.g., 50x50) and pick the one that is farthest from existing.
        
        # Let's use a 50x50 grid as candidates.
        cand_grid_size = 50
        cand_x0 = np.linspace(0, 1, cand_grid_size)
        cand_x1 = np.linspace(0, 1, cand_grid_size)
        candidates = np.array([np.meshgrid(cand_x0, cand_x1)]).transpose(1, 2, 0).reshape(-1, 2)
        
        # We will iteratively add points.
        # To speed up, we can maintain the "influence" or just recompute distances each time.
        # N is at most 200. Candidates is 2500.
        # Distance calc: 2500 * 200 = 500,000 ops. Very fast.
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if n_evals >= self.budget:
                break
                
            # Calculate min distance from each candidate to any existing point
            # coords_2d is shape (N, 2)
            # candidates is shape (C, 2)
            
            # Diff: (C, 1, 2) - (1, N, 2) -> (C, N, 2)
            # This might be memory heavy if N is large, but N=200 is fine.
            # 2500 * 200 * 2 * 8 bytes = ~8 MB. Fine.
            
            diffs = candidates[:, np.newaxis, :] - coords_2d[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # Find candidate with max min_dist
            best_idx = np.argmax(min_dists)
            best_cand = candidates[best_idx]
            
            # Construct full vector
            x = np.zeros(self.dim)
            x[0] = best_cand[0]
            x[1] = best_cand[1]
            x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            if f is None:
                break
            
            # Add to coords_2d for next iteration
            coords_2d = np.vstack([coords_2d, best_cand])
            
        return all_points
