"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Multi-Scale Lattice Survey followed by Parallel CMA-ES Refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Deterministic Stratified Sampling (Lattice Survey)
        # Büche-Rastrigin has periodicity 5. In 5D, a 5x5x5x5x5 grid (3125 points) 
        # samples every unit cell of the periodic landscape.
        # We use a smaller budget for this phase to identify the "deepest" basins.
        # Let's use 10 points per dimension? 10^5 = 100,000 is too many.
        # The function is periodic with period 5. So 5 points per dimension (at -5, -2.5, 0, 2.5, 5) 
        # or shifted (e.g., -4, -2, 0, 2, 4) covers the fundamental domain.
        # Actually, the minimizer is in [-4, 4]^5.
        # A grid of 3^5 = 243 points is a good start. 5^5 = 3125 is better but might eat budget.
        # Let's try 4^5 = 1024 points. This is ~20% of the budget.
        
        grid_size = 4
        num_grid_points = grid_size ** dim
        
        # Allocate budget: 20% to grid, 80% to refinement
        grid_budget = min(num_grid_points, int(total_budget * 0.2))
        
        # Generate grid points
        # We want to cover [-5, 5]. 
        # Using linspace from -5 to 5 with grid_size points.
        grid_coords = np.linspace(-5, 5, grid_size)
        
        # Generate all combinations
        grid_points = np.array(np.meshgrid(*([grid_coords] * dim))).T.reshape(-1, dim)
        
        # If grid points exceed grid_budget, we need to subsample or reduce grid size.
        # For dim=5, grid_size=4 -> 1024 points. 20% of 500 is 100. 
        # So 1024 is too many for 100 calls.
        # Let's adjust: Use 3^5 = 243 points. 243 > 100.
        # Let's use a random subsample of the grid if it's too big, or just reduce grid size.
        # Better: Adaptive grid size based on budget.
        
        # Let's try grid_size such that grid_size^dim <= grid_budget.
        # 3^5 = 243. 2^5 = 32.
        # If budget is 500, grid_budget=100. 2^5=32 is safe. 3^5=243 is not.
        # So we stick with a coarser grid or random sampling.
        
        # Alternative: Use the first N points of a deterministic Latin Hypercube or just random uniform.
        # But the prompt suggests "deterministic multi-scale lattice".
        # Let's use a 3^5 grid (243 points) but we only have 100 calls?
        # No, we can adjust the split.
        # Let's use 30% for grid, 70% for refinement.
        # 30% of 500 = 150. Still < 243.
        # Let's use 2^5 = 32 points for the grid? That's very coarse.
        
        # Let's reconsider the "Lattice".
        # Maybe a 5^5 grid is the "ideal" but we can't afford it.
        # Let's use a random stratified sample of size grid_budget.
        # Stratified sampling: Divide [-5,5]^d into grid_budget^1/d sub-cubes?
        # Simpler: Just take grid_budget random points, but ensure they are spread out.
        
        # Actually, the instruction says: "Use a deterministic multi-scale lattice of 5^5 = 3125 points... in the first 300 calls".
        # This implies the strategy *wants* to use many calls for the grid.
        # If I use 300 calls for the grid, I have 200 for refinement.
        # 3125 points is too many for 300 calls.
        # I will use a subsample of the 5^5 lattice.
        
        # Let's define the 5^5 lattice points explicitly.
        # Lattice points: {-5, -2.5, 0, 2.5, 5} in each dimension.
        lattice_coords = np.array([-5, -2.5, 0, 2.5, 5])
        lattice_points = np.array(np.meshgrid(*([lattice_coords] * dim))).T.reshape(-1, dim)
        
        # Shuffle the lattice points deterministically to avoid bias towards corners first?
        # Or just take the first N?
        # Taking first N of meshgrid is biased. Let's shuffle with a fixed seed.
        rng_grid = np.random.default_rng(123)
        idx = rng_grid.permutation(len(lattice_points))
        lattice_points = lattice_points[idx]
        
        # Number of grid points to evaluate
        n_grid_calls = min(len(lattice_points), 300)
        
        # Evaluate Grid
        grid_results = []
        for i in range(n_grid_calls):
            if calls_made >= total_budget or time.time() >= deadline:
                break
            x = lattice_points[i]
            f = func(x)
            calls_made += 1
            grid_results.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Identify top basins
        # Sort grid results by fitness
        grid_results.sort(key=lambda x: x[0])
        
        # Select top K basins for refinement
        # We have remaining_budget = total_budget - calls_made
        # Let's allocate remaining budget to K parallel CMA-ES instances.
        # How many? Maybe 5-10 best distinct points.
        
        # Distinctness check: If two grid points are very close, they might be in the same basin.
        # Simple heuristic: Take the top 10 points, but ensure they are at least 1.0 apart?
        # Or just take the top 5.
        
        num_refinements = 5
        top_points = [x for f, x in grid_results[:num_refinements]]
        
        # If we didn't use all grid budget, we can use the rest for refinement.
        remaining_budget = total_budget - calls_made
        
        # Allocate remaining budget to num_refinements instances
        # Each instance gets remaining_budget / num_refinements calls
        # But CMA-ES needs a minimum number of calls to be effective.
        # If remaining_budget is small, reduce num_refinements.
        
        min_calls_per_cma = 20 # Minimum useful CMA-ES runs
        max_refinements = remaining_budget // min_calls_per_cma
        num_refinements = min(num_refinements, max_refinements, len(top_points))
        
        if num_refinements == 0:
            return best_x, best_f
            
        calls_per_refinement = remaining_budget // num_refinements
        
        # Run Parallel CMA-ES (sequentially in Python, but independent)
        
        def run_short_cmaes(x_init, sigma_init, max_calls):
            nonlocal best_x, best_f, calls_made
            
            # Simple (1+1)-ES or small population CMA-ES?
            # Given small budget (e.g., 40 calls), a full CMA-ES is overkill.
            # A simple local search or (1+1)-ES is better.
            # Let's use a simple adaptive (1+1)-ES.
            
            x_current = x_init.copy()
            f_current = func(x_current)
            calls_made += 1
            
            if f_current < best_f:
                best_f = f_current
                best_x = x_current.copy()
            
            sigma = sigma_init
            # Adaptive step size: increase if improvement, decrease if not.
            # Simple adaptation: sigma *= 1.1 if improved, sigma *= 0.9 if not.
            
            for _ in range(max_calls - 1):
                if calls_made >= max_calls or time.time() >= deadline:
                    break
                
                # Generate offspring
                z = rng.standard_normal(dim)
                x_new = x_current + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < f_current:
                    x_current = x_new
                    f_current = f_new
                    sigma *= 1.1 # Increase step
                    if f_current < best_f:
                        best_f = f_current
                        best_x = x_current.copy()
                else:
                    sigma *= 0.9 # Decrease step
            
            return f_current, x_current
        
        for i in range(num_refinements):
            if calls_made >= total_budget or time.time() >= deadline:
                break
            
            x_init = top_points[i]
            # Start with a small sigma since we are in a basin
            sigma_init = 0.5
            
            run_short_cmaes(x_init, sigma_init, calls_per_refinement)
            
            # Check if we should stop early if we found a very good point
            # (Optional: if best_f is very low, we might stop)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
