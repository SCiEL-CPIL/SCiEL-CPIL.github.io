"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using deterministic Sobol' sequence sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Use a Sobol' sequence to generate a low-discrepancy set of points.
        # This provides much better coverage of the 4D space than random sampling,
        # especially with a small budget like 200.
        # We will generate 'budget' points using Sobol' sampling.
        
        # Sobol' requires the number of points to be a power of 2.
        # Find the largest power of 2 <= budget
        n_points = 1
        while n_points * 2 <= self.budget:
            n_points *= 2
            
        # Generate Sobol' points in [0, 1]^dim
        # qmc.Sobol requires dim <= 21
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        # We need n_points points. 
        # Note: Sobol generates points in [0,1].
        # We use 'scramble=True' to avoid correlation with the grid structure.
        
        # Generate the points
        # The method 'random' generates n points.
        # We ensure n_points is a power of 2.
        sobol_points = sampler.random(n_points)
        
        # Map to bounds [lo, hi]
        x_samples = lo + (hi - lo) * sobol_points
        
        # Evaluate all points
        # We record all evaluated points.
        # The task says "Every returned objective vector is recorded".
        # We just need to call problem(x) for each x.
        
        # We might have budget > n_points (e.g., budget=200, n_points=128).
        # We can use the remaining budget for a few local refinements or just stop.
        # Given the stagnation directive and the simplicity of the problem (4 vars, 3 objs),
        # uniform coverage is likely the best use of the budget to maximize hypervolume.
        # However, if we have leftover budget, we can use it to refine the best non-dominated points.
        # But for now, let's stick to the pure sampling strategy as it's structurally different 
        # and robust. If we have time, we can add a simple local search on the best points.
        
        # Let's evaluate the Sobol points first.
        count = 0
        for i in range(n_points):
            if time.time() - start_time > max_time:
                break
            x = x_samples[i]
            problem(x)
            count += 1
            
        # If we have budget left, use it for a simple local search around the best non-dominated points found so far?
        # Or just generate more random points?
        # Sobol is deterministic. Let's just fill the rest with random points if any budget remains.
        # Actually, Sobol with n=128 covers the space well. 
        # Let's use the remaining 72 evaluations to do a simple random perturbation around the 
        # points that are currently non-dominated.
        
        # To do this, we need to keep track of the points and their objectives.
        # But the problem interface doesn't return the points, only the objectives are recorded externally.
        # Wait, the problem statement says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # It also says: "problem(x) ... returns 3 objective values."
        # So I don't see the objectives in my code? 
        # "problem(x) clips ... and returns 3 objective values."
        # Yes, I can access the return value.
        
        # Let's restart the logic to properly capture the objectives for potential refinement.
        
        # Reset logic
        count = 0
        x_best = []
        f_best = []
        
        # 1. Evaluate Sobol points
        for i in range(n_points):
            if time.time() - start_time > max_time:
                break
            x = x_samples[i]
            f = problem(x)
            x_best.append(x)
            f_best.append(f)
            count += 1
            
        # 2. Use remaining budget for local search around non-dominated points
        if count < self.budget:
            # Identify non-dominated points in the current set
            # f_best is a list of arrays
            f_arr = np.array(f_best)
            n_found = len(f_arr)
            
            # Simple non-dominated check
            # A point i is non-dominated if there is no j such that f_j <= f_i and f_j != f_i
            # With small n_found, O(n^2) is fine.
            non_dominated_indices = []
            for i in range(n_found):
                dominated = False
                for j in range(n_found):
                    if i == j:
                        continue
                    if np.all(f_arr[j] <= f_arr[i]) and np.any(f_arr[j] < f_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated_indices.append(i)
            
            # If we have non-dominated points, pick a few to refine
            # We have budget - count evaluations left.
            # Let's distribute them among the top few non-dominated points.
            n_refine = min(len(non_dominated_indices), 5) # Refine at most 5 points
            
            if n_refine > 0:
                # Pick the first n_refine non-dominated points
                indices_to_refine = non_dominated_indices[:n_refine]
                evals_per_refine = (self.budget - count) // n_refine
                if evals_per_refine == 0:
                    evals_per_refine = 1
                    n_refine = self.budget - count
                
                # For each selected point, do a small random walk
                rng = np.random.default_rng(123)
                for idx in indices_to_refine:
                    if count >= self.budget:
                        break
                    if time.time() - start_time > max_time:
                        break
                    
                    x_current = x_best[idx]
                    f_current = f_best[idx]
                    
                    for _ in range(evals_per_refine):
                        if count >= self.budget:
                            break
                        if time.time() - start_time > max_time:
                            break
                        
                        # Small random perturbation
                        step_size = 0.05 * (hi - lo)
                        perturbation = rng.standard_normal(self.dim) * step_size
                        x_new = np.clip(x_current + perturbation, lo, hi)
                        
                        f_new = problem(x_new)
                        count += 1
                        
                        # If it improves the current best for this local search (simple greedy), update x_current
                        # Since we are just trying to find better points, we can check if it's better than f_current
                        # in a weighted sum sense or just keep the best non-dominated?
                        # Let's just accept if it's better in at least one objective and not worse in others?
                        # Or simpler: just accept if the sum is lower?
                        # Let's use a simple hypervolume contribution proxy: sum of objectives.
                        if np.sum(f_new) < np.sum(f_current):
                            x_current = x_new
                            f_current = f_new
                            
        return None

# EVOLVE-BLOCK-END
