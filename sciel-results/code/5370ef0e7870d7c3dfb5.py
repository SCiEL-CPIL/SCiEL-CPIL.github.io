import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse Grid Search (3^5 = 243 points)
        # This covers the entire domain [-5, 5]^5 with a 3x3x3x3x3 grid.
        # Step size is 10/2 = 5. Points are at -5, 0, 5 in each dimension.
        # Wait, for 3 points, the indices are 0, 1, 2.
        # x_i = -5 + i * (10 / (3-1)) = -5, 0, 5.
        # This is a very coarse grid. It ensures we don't miss the global minimum
        # if the function's global minimum is not extremely sharp or if the shift
        # keeps it within a cell. Given the fractal nature, the global minimum
        # is likely unique or one of the deep wells. A 3^5 grid is cheap (243 evals)
        # and provides a robust initial estimate.
        
        levels = [3, 5, 7, 9]
        # We will perform a hierarchical refinement.
        # Start with the full domain.
        
        # Initialize the current search box to the full domain
        current_low = np.full(dim, -5.0)
        current_high = np.full(dim, 5.0)
        
        # We will iterate through the grid levels.
        # For each level, we evaluate a grid within the current box.
        # We then shrink the box to the best point's neighborhood.
        
        # However, a pure hierarchical shrink might get stuck if the coarse grid
        # misses the minimum. But for Katsuura, the global minimum is typically
        # at a specific fractal point. The coarse grid of 243 points is likely
        # sufficient to identify the correct region, especially since the function
        # is bounded and the minimum is in [-4, 4]^5.
        
        # Let's refine the strategy:
        # 1. Evaluate 3^5 grid on full domain.
        # 2. Find best point.
        # 3. Shrink domain to a box around the best point (e.g., +/- half the grid step).
        # 4. Evaluate 5^5 grid on this smaller domain.
        # 5. Find best point.
        # 6. Shrink domain again.
        # 7. Evaluate 7^7 grid? No, 7^5 = 16807 which is too many.
        #    Budget is 500.
        #    3^5 = 243. Remaining: 257.
        #    5^5 = 3125. Too many.
        #    So we can only do one refinement level if we use 5^5? No, 3125 > 257.
        #    We need smaller grids or fewer dimensions? No, dim is fixed.
        
        # Revised Budget Allocation:
        # Phase 1: 3^5 = 243 evals.
        # Phase 2: We have ~257 evals left.
        #    We can do a 4^5 = 1024 grid? No.
        #    We can do a 3^5 grid again on the refined box? That's 243 evals.
        #    Total 486. This leaves 14 evals for a final tweak.
        #    Alternatively, a 2^5 = 32 grid? Too coarse.
        #    What if we do a 3^5 grid, then a 3^5 grid on the best sub-box?
        #    The sub-box size would be 1/3 of the original.
        #    Step 1: Grid step 5.0. Box size 10.0.
        #    Step 2: Best point found. New box: best_point +/- 2.5. Box size 5.0.
        #    Step 3: Grid 3^5 on new box. Step 2.5.
        #    Step 4: Best point. New box: best_point +/- 1.25. Box size 2.5.
        #    Step 5: Grid 3^5 on new box. Step 1.25.
        #    Total evals: 243 * 3 = 729. Too many.
        
        # Let's stick to the directive's suggestion: "coarse 3^5 grid ... then recursively refining ... (e.g., 5^5, then 7^5)"
        # But 5^5 is 3125. This exceeds the budget.
        # The directive might imply that we *try* these but are constrained by budget.
        # Or maybe the "5^5" refers to the number of points per dimension?
        # If we use a 5-point grid in 5 dimensions, that's 5^5 = 3125.
        # This is impossible with budget 500.
        
        # Perhaps the strategy is:
        # 1. Coarse 3^5 grid (243 evals).
        # 2. Identify the best cell.
        # 3. Refine that specific cell with a finer grid, but limited by remaining budget.
        #    Remaining budget: ~257.
        #    If we refine the best cell (which is 1/3 the size of the domain in each dim),
        #    we can use a 4^5 grid? 1024 > 257.
        #    We can use a 3^5 grid on the best cell? 243 <= 257.
        #    This gives a total of 486 evals.
        #    The resolution improves by a factor of 3 in each step.
        #    Step 1: Resolution 5.0.
        #    Step 2: Resolution 5.0/3 = 1.67.
        #    This is a significant improvement.
        
        # Let's implement this:
        # 1. Grid 3^5 on [-5, 5]^5.
        # 2. Find best point x_best.
        # 3. Define new box: [x_best - step/2, x_best + step/2] where step = 10/2 = 5.
        #    So new box is [x_best - 2.5, x_best + 2.5].
        #    Clip to [-5, 5] if necessary (though x_best is inside).
        # 4. Grid 3^5 on this new box.
        # 5. Find best point.
        # 6. Use remaining budget for a few random perturbations or a very fine local search?
        #    Or just stop. 486 evals used. 14 left.
        #    We can use the remaining 14 evals to do a simple coordinate descent or
        #    just evaluate the center of the final box.
        
        # Let's refine the grid generation to be efficient.
        
        def generate_grid(low, high, n_points_per_dim):
            # Generate n_points_per_dim points in each dimension
            # Returns a list of points
            points = []
            # Use a recursive or iterative approach
            # For 5 dimensions, 3^5 = 243 points.
            # We can use np.meshgrid
            grids = np.meshgrid(*[np.linspace(low[i], high[i], n_points_per_dim) for i in range(dim)])
            # Flatten
            flat_grids = [g.flatten() for g in grids]
            points = np.column_stack(flat_grids)
            return points

        # Phase 1: Coarse Grid
        n1 = 3
        points1 = generate_grid(current_low, current_high, n1)
        for p in points1:
            if eval_count >= budget or time.time() > deadline - 5:
                break
            counting_func(p)
        
        if eval_count >= budget or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: Refine around best
        # Calculate step size of the coarse grid
        step1 = (current_high[0] - current_low[0]) / (n1 - 1)
        
        # New box centered at best_x, size step1 in each dimension
        # Actually, the "cell" containing best_x is of size step1.
        # The center of the cell is best_x.
        # So the box is best_x +/- step1/2.
        new_low = best_x - step1 / 2.0
        new_high = best_x + step1 / 2.0
        
        # Clip to domain
        new_low = np.clip(new_low, -5.0, 5.0)
        new_high = np.clip(new_high, -5.0, 5.0)
        
        # Ensure low <= high
        for i in range(dim):
            if new_low[i] > new_high[i]:
                new_low[i], new_high[i] = new_high[i], new_low[i]
        
        # Evaluate 3^5 grid on new box
        n2 = 3
        points2 = generate_grid(new_low, new_high, n2)
        for p in points2:
            if eval_count >= budget or time.time() > deadline - 5:
                break
            counting_func(p)
            
        if eval_count >= budget or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 3: Further Refine
        step2 = (new_high[0] - new_low[0]) / (n2 - 1)
        
        new_low2 = best_x - step2 / 2.0
        new_high2 = best_x + step2 / 2.0
        
        new_low2 = np.clip(new_low2, -5.0, 5.0)
        new_high2 = np.clip(new_high2, -5.0, 5.0)
        
        for i in range(dim):
            if new_low2[i] > new_high2[i]:
                new_low2[i], new_high2[i] = new_high2[i], new_low2[i]
                
        # We have ~14 evals left.
        # A 3^5 grid would be 243 evals. Too many.
        # We can do a 2^5 = 32 grid? No, 32 > 14.
        # We can do a 2^4 = 16 grid? But dim is 5.
        # We can just evaluate the center of the box and maybe 4 neighbors?
        # Or just stop. The hierarchical grid has already reduced the search space
        # by a factor of 9 in each dimension (3 * 3).
        # Resolution went from 5.0 to 1.67.
        
        # Let's use the remaining budget for a simple local refinement:
        # Evaluate the center of the final box (which is best_x).
        # Then evaluate points slightly offset in each dimension.
        
        # Actually, let's just evaluate the best_x again? No, it's already evaluated.
        # Let's try a few random points in the final box to see if we can improve.
        # Or just return.
        
        # To be safe and maximize score, let's use remaining budget for a tiny
        # Nelder-Mead or just random sampling in the final box.
        # Given the small budget, random sampling in the final box is sufficient.
        
        remaining = budget - eval_count
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                # Sample in the final box
                p = rng.uniform(new_low2, new_high2)
                counting_func(p)

        return best_x, best_f
