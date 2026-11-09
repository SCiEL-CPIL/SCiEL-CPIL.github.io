"""Multi-objective optimizer on WFG4 using Grid-based Decomposition with Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)
            
        # Strategy:
        # 1. Partition the search space into a grid of cells.
        # 2. Use Latin Hypercube Sampling (LHS) to place one initial point in each cell.
        #    This ensures maximal spatial coverage of the 12D space.
        # 3. For each cell, perform a local search (Nelder-Mead) to converge to a local Pareto optimum.
        #    The local search minimizes a scalarized objective (weighted sum) derived from the cell's center
        #    to guide the search towards a specific region of the Pareto front.
        # 4. Record all evaluated points. The union of local optima from different cells should cover the front well.

        # Grid Parameters
        # We have 200 evaluations.
        # Let's use a grid of 5x5x5 = 125 cells? Or 4x4x4 = 64 cells?
        # With 12 dimensions, a full grid of 2^12 = 4096 is too many.
        # A sparse grid or LHS is better.
        # Let's use N_START = 50 starting points.
        # Each local search uses ~3-5 evaluations if we use a simple gradient-free method or a few Nelder-Mead steps.
        # Actually, Nelder-Mead in 12D might take many iterations.
        # Let's use a simpler local search: Coordinate Descent or a few steps of Powell.
        # Or even simpler: a small population-based local search around each seed.
        
        # Let's try:
        # 1. Generate 40 seed points using LHS in 12D.
        # 2. For each seed, run a small local search with a budget of 4 evaluations.
        #    Total evaluations: 40 * (1 + 4) = 200.
        #    The local search will try to improve the seed point.
        
        n_starts = 40
        evals_per_search = 4  # Including the initial evaluation? Let's say 1 initial + 4 steps = 5.
        # 40 * 5 = 200. Perfect.
        
        # Generate LHS points
        # LHS in 12D: For each dimension, divide [0,1] into n_starts intervals and pick one random point in each.
        # Then shuffle the intervals for each dimension independently.
        
        def lhs(n, dim):
            samples = np.zeros((n, dim))
            for d in range(dim):
                # Create n intervals [0, 1/n), [1/n, 2/n), ...
                # Pick a random point in each
                pts = (np.arange(n) + np.random.rand(n)) / n
                # Shuffle
                np.random.shuffle(pts)
                samples[:, d] = pts
            return samples
        
        lhs_points = lhs(n_starts, dim)
        
        # Map to bounds
        seeds = lo + lhs_points * span
        
        used = 0
        all_points = []
        
        # Local Search Strategy:
        # For each seed, we will perform a simple local search.
        # We'll use a weighted sum scalarization with weights derived from the seed's position in the LHS space.
        # Actually, a simpler approach: just do a random walk with hill climbing on the hypervolume contribution?
        # No, that's complex.
        # Let's use Nelder-Mead from scipy.optimize.
        # But we need to minimize a single objective.
        # We can minimize a weighted sum: sum(w_i * f_i(x)).
        # How to choose weights?
        # If we use uniform weights, we'll converge to the same point (the knee point).
        # To cover the front, we need different weights for each start.
        # We can generate random weight vectors.
        
        # Let's generate random weight vectors for each start.
        weights = np.random.dirichlet(np.ones(n_obj), size=n_starts)
        
        # Define the scalarized objective
        def scalarized_objective(x, w):
            f = problem(x)
            # We need to track evaluations
            nonlocal used
            # Note: problem(x) is called here.
            # We must be careful with the budget.
            # Nelder-Mead will call this function multiple times.
            # We need to stop if we exceed budget or time.
            
            # Since we can't easily interrupt Nelder-Mead mid-call without a callback,
            # we will wrap the function to count evaluations and raise an exception or return a large value if budget is exceeded.
            # However, the task says "exception before the first makes result invalid".
            # So we must avoid exceptions.
            # We can just return a large penalty if budget is exceeded, but NM might keep going.
            # Better: use a custom local search that we control.
            
            # Let's implement a simple local search:
            # 1. Evaluate seed.
            # 2. Generate neighbors by perturbing the seed.
            # 3. If a neighbor is better (in terms of weighted sum), move to it.
            # 4. Repeat for a few steps.
            
            pass # This function is not used directly in the loop below, just for conceptual clarity.

        # Let's implement the local search loop manually to control budget strictly.
        
        for i in range(n_starts):
            if used >= budget or time.time() > deadline:
                break
                
            x = seeds[i]
            w = weights[i]
            
            # 1. Evaluate seed
            f_seed = problem(x)
            used += 1
            all_points.append((x, f_seed))
            
            if used >= budget or time.time() > deadline:
                break
            
            # 2. Local Search
            # We will try to improve x by minimizing w @ f(x).
            # We use a simple random direction hill climbing.
            # Step size: start with 0.1 * span, decrease if no improvement.
            
            current_x = x
            current_f = f_seed
            current_score = np.dot(w, current_f)
            
            step_size = 0.1
            no_improve_count = 0
            max_steps = evals_per_search - 1 # We already used 1 eval for seed
            
            for step in range(max_steps):
                if used >= budget or time.time() > deadline:
                    break
                
                # Generate a random direction
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction)
                
                # Try a step in this direction
                new_x = current_x + step_size * direction
                # Clip to bounds
                new_x = np.clip(new_x, lo, hi)
                
                # Evaluate
                new_f = problem(new_x)
                used += 1
                all_points.append((new_x, new_f))
                
                new_score = np.dot(w, new_f)
                
                if new_score < current_score:
                    # Improvement
                    current_x = new_x
                    current_f = new_f
                    current_score = new_score
                    no_improve_count = 0
                    # Maybe increase step size slightly? Or keep it.
                else:
                    # No improvement
                    no_improve_count += 1
                    step_size *= 0.5 # Shrink step size
                    if step_size < 1e-6:
                        break

        # The evaluator records all returned objective vectors.
        # We have stored them in all_points, but the evaluator wraps problem() so it records automatically.
        # We just need to return something.
        
        return None

# EVOLVE-BLOCK-END
