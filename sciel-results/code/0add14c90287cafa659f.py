"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Structured Grid Search on a 3D Manifold
        # WFG9's Pareto front is primarily determined by the first 3 decision variables.
        # The remaining 9 variables can be fixed (e.g., to midpoints) without losing 
        # Pareto optimality in the objective space, allowing us to reduce the search 
        # space from 12D to 3D and use the budget for a dense grid.
        
        # Fix variables 3..11 to their midpoints
        mid = (lo + hi) / 2.0
        fixed_tail = mid[3:]
        
        # We will search over x[0], x[1], x[2]
        # Total budget is 200.
        # A 3D grid with N points per axis has N^3 points.
        # 5^3 = 125. 6^3 = 216 (too many).
        # Let's use a 5x5x5 grid for the first pass (125 evals).
        # Remaining budget: 200 - 125 = 75.
        # We can use the remaining budget for a finer grid or local refinement.
        
        # However, to maximize hypervolume, a uniform grid is often good, but 
        # a slightly non-uniform or adaptive one might be better. 
        # Given the "deterministic" hint, let's stick to a robust grid.
        
        # Let's try a 5x5x5 grid first.
        N = 5
        grid_size = N ** 3
        if grid_size > self.budget:
             # Reduce N if budget is small
             N = int(self.budget ** (1.0/3.0))
             grid_size = N ** 3
             
        # Generate grid points for the first 3 dimensions
        # np.linspace includes endpoints. 
        # It's often better to avoid exact boundaries if the function is singular, 
        # but WFG9 is well-behaved. Let's use linspace.
        
        x0_vals = np.linspace(lo[0], hi[0], N)
        x1_vals = np.linspace(lo[1], hi[1], N)
        x2_vals = np.linspace(lo[2], hi[2], N)
        
        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                for x2 in x2_vals:
                    x = np.empty(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    x[3:] = fixed_tail
                    points.append(x)
        
        # Evaluate all grid points
        # Ensure we don't exceed budget
        points = points[:self.budget]
        
        F = np.array([problem(x) for x in points])
        used = len(points)
        
        # If we have budget left, we can perform a second, finer grid on a subset
        # or just add more points to the grid. 
        # With N=5, we used 125. We have 75 left.
        # We could do a second grid with N=4 (64 points) on a sub-region?
        # Or simply increase N to 5 and use the remaining budget to add points 
        # to the corners or mid-edges?
        
        # A simpler approach: Just use the full budget for a slightly larger grid 
        # if possible, or stick to 5x5x5 and use remaining for random sampling?
        # The inspiration says "deterministic, structured grid search". 
        # Let's stick to the 5x5x5 grid for the core, and if budget allows, 
        # add a second layer of points?
        
        # Actually, 6x6x6 = 216 > 200. 
        # 5x5x5 = 125.
        # We have 75 points left.
        # We can add a 4x4x4 grid (64 points) shifted by half a step?
        # Or just evaluate 75 random points? 
        # The prompt says "deterministic". So random is out.
        
        # Let's add a second grid of size 4x4x4 = 64 points, offset by half the step size.
        # This effectively doubles the resolution in some areas.
        if used < self.budget and time.time() < deadline:
            N2 = 4
            grid_size2 = N2 ** 3
            if used + grid_size2 <= self.budget:
                # Offset by half step
                step0 = (hi[0] - lo[0]) / N
                step1 = (hi[1] - lo[1]) / N
                step2 = (hi[2] - lo[2]) / N
                
                # The first grid was at lo + i*step for i in 0..N-1? 
                # np.linspace(lo, hi, N) gives points at lo, lo + (hi-lo)/(N-1), ...
                # So the step is (hi-lo)/(N-1).
                # Let's use the same linspace logic but offset.
                
                x0_vals2 = np.linspace(lo[0] + step0/2, hi[0] - step0/2, N2)
                x1_vals2 = np.linspace(lo[1] + step1/2, hi[1] - step1/2, N2)
                x2_vals2 = np.linspace(lo[2] + step2/2, hi[2] - step2/2, N2)
                
                new_points = []
                for x0 in x0_vals2:
                    for x1 in x1_vals2:
                        for x2 in x2_vals2:
                            x = np.empty(self.dim)
                            x[0] = x0
                            x[1] = x1
                            x[2] = x2
                            x[3:] = fixed_tail
                            new_points.append(x)
                            
                # Check budget again
                max_new = self.budget - used
                new_points = new_points[:max_new]
                
                if new_points:
                    F2 = np.array([problem(x) for x in new_points])
                    F = np.vstack((F, F2))
                    used += len(new_points)
        
        # The return value of __call__ is ignored, but we should return something valid
        # The task says "Every returned objective vector is recorded". 
        # Wait, "Every returned objective vector is recorded" usually implies the evaluator 
        # records the outputs of the `problem` calls? 
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This likely means the evaluator 
        # wraps the `problem` function to record every call's output.
        # "and the return value of __call__ is ignored."
        # So we just need to make sure we call `problem` exactly the number of times 
        # we want to record, and we don't exceed the budget.
        
        # We have called problem `used` times.
        # We should return the population or just None. The contract says return value is ignored.
        # Let's return the last evaluated points or just None.
        return None

# EVOLVE-BLOCK-END
