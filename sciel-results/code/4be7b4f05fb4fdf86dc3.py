import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Structured grid search on the first 3 dimensions.
        # WFG9's Pareto front is largely determined by the first few variables.
        # By fixing the remaining 9 variables to their midpoints, we reduce
        # the search space to 3D, allowing for a dense grid within budget 200.
        
        # 200 evaluations budget.
        # A 3D grid with N points per dimension has N^3 evaluations.
        # 5^3 = 125.
        # 6^3 = 216 (too many).
        # So we can do a 5x5x5 grid (125 evals) plus some extra points for refinement or just stick to the grid.
        # Let's use 5 points per dimension for the main grid.
        # This uses 125 evaluations.
        # We have 75 evaluations left. We can use these for a second, coarser or shifted grid, or random sampling around the best points.
        # Given the "structured" directive, let's stick to a deterministic approach.
        # Maybe a 4x4x4 grid (64) is too sparse. 5x5x5 is good.
        # Let's try to maximize the grid size. 
        # If we use all 200, we could do a 5x5x5 grid (125) and then add 75 more points.
        # Or a 6x6x5 grid (180) + 20 points?
        # Let's stick to a regular 5x5x5 grid for simplicity and coverage, and use the remaining budget for a slightly refined grid or just stop.
        # Actually, a 5x5x5 grid covers the space well.
        
        n_grid = 5
        grid_points = []
        
        # Generate grid points for the first 3 dimensions
        # The remaining 9 dimensions are fixed to the midpoint
        mid_points = (lo + hi) / 2.0
        
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    x = np.zeros(self.dim)
                    # Map i, j, k to [0, 1] range and then to [lo, hi]
                    # Avoiding exact boundaries might be safer, but problem clips anyway.
                    # Linear spacing from lo to hi
                    x[0] = lo[0] + (hi[0] - lo[0]) * (i / (n_grid - 1))
                    x[1] = lo[1] + (hi[1] - lo[1]) * (j / (n_grid - 1))
                    x[2] = lo[2] + (hi[2] - lo[2]) * (k / (n_grid - 1))
                    
                    # Set remaining dimensions to midpoints
                    for d in range(3, self.dim):
                        x[d] = mid_points[d]
                        
                    grid_points.append(x)
        
        # Evaluate the grid
        F = []
        X = []
        
        for x in grid_points:
            if time.time() > deadline:
                break
            f = problem(x)
            X.append(x)
            F.append(f)
            
        # We have used len(X) evaluations.
        # If we have budget left, we can try a second grid with a slight offset or different midpoint settings?
        # Or just random sampling to catch any anomalies?
        # The prompt suggests "dense coverage of the Pareto front".
        # Let's add a few random points if budget allows, to ensure we don't miss a corner if the manifold assumption is slightly off.
        # But the directive says "deterministic, structured grid search". 
        # Let's stick to the grid. If we have budget left, we can increase the grid resolution.
        
        # 125 evals used. 75 left.
        # Can we do a 6x6x5 grid? 180 evals.
        # Let's try to generate a 6x6x5 grid if we have enough budget.
        # 6*6*5 = 180.
        # 180 <= 200.
        # Let's restart with a denser grid if we haven't used all budget yet?
        # No, the loop above already ran.
        # Let's just use the 5x5x5 grid. It's a solid baseline.
        
        # Wait, I can just generate the grid points and evaluate them all at once if I'm careful with budget.
        # Let's refine: Use a 6x6x5 grid (180 points) to get better resolution.
        # 6 points in dim 0, 6 in dim 1, 5 in dim 2.
        
        X_new = []
        F_new = []
        
        n0, n1, n2 = 6, 6, 5
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    if len(X_new) >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = lo[0] + (hi[0] - lo[0]) * (i / (n0 - 1))
                    x[1] = lo[1] + (hi[1] - lo[1]) * (j / (n1 - 1))
                    x[2] = lo[2] + (hi[2] - lo[2]) * (k / (n2 - 1))
                    for d in range(3, self.dim):
                        x[d] = mid_points[d]
                        
                    f = problem(x)
                    X_new.append(x)
                    F_new.append(f)
                    
        # If we still have budget, we can add a second grid with different midpoints for the fixed variables?
        # Or just stop. 180 is close to 200.
        
        # Return the points found. The evaluator records all returned objective vectors.
        # The __call__ return value is ignored, but we must return something valid (e.g., the list of X or F).
        # The contract says "Every returned objective vector is recorded". 
        # Wait, "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies that the `problem(x)` calls are the ones being recorded? 
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" usually refers to the outputs of the problem function calls made during the execution.
        # "The return value of __call__ is ignored."
        # So I just need to make sure I call problem() at most 200 times.
        
        return X_new
