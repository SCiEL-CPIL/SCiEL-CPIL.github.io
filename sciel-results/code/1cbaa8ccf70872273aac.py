"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Multi-objective optimizer using Grid-based Local Search (Nelder-Mead).
    Strategy:
    1. Sample a small number of random starting points.
    2. For each starting point, perform local optimization (Nelder-Mead) 
       minimizing a scalarized objective function (weighted sum or Tchebycheff).
    3. Maintain an external archive of non-dominated solutions found during local searches.
    
    This approach is better suited for WFG4's deceptive landscape than DE because
    it focuses computational effort on refining promising regions rather than 
    maintaining a broad but shallow population.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # We will use a "Local Search" strategy.
        # 1. Pick N_start starting points.
        # 2. For each, run Nelder-Mead for a limited number of iterations.
        
        # WFG4 has 12 dimensions. Nelder-Mead is expensive (dim+1 simplex points).
        # Each iteration of NM requires evaluating the worst point's replacement.
        # Let's estimate cost: 1 eval per iteration roughly (approx).
        # Budget 200.
        # If we use 10 starting points, we have 20 evals each.
        # NM in 12D with 20 evals is very short, but might find local optima if lucky.
        # Alternative: Use fewer starting points but longer searches?
        # Or use a hybrid: Random search first to find good regions, then local search?
        
        # Let's try: 
        # 1. Random sample 10 points.
        # 2. Select the best 5 based on a scalarized metric (e.g., distance to ideal point).
        # 3. Run Nelder-Mead from these 5 points, allocating remaining budget.
        
        # Actually, WFG4 is deceptive. The "ideal point" (0,0,0) is not the goal for all objectives.
        # We want to minimize all 3.
        # Let's use a simple weighted sum for the local search target.
        # But weights matter. We can use multiple weight vectors.
        
        # Strategy:
        # 1. Initialize with 10 random points.
        # 2. Identify the "best" 3-5 points using a hypervolume contribution proxy or simple sum.
        # 3. For each of these, run a local Nelder-Mead optimization minimizing 
        #    a specific weighted sum of objectives.
        
        # However, implementing a robust Nelder-Mead from scratch with budget tracking 
        # and bounds clipping is tricky. 
        # Let's use a simpler local search: Coordinate Descent or a simple pattern search.
        # Or, since SciPy is available, we could use scipy.optimize.minimize, 
        # but we must count every function call to `problem`.
        # scipy.optimize.minimize calls the function many times. We need to wrap it.
        
        # Let's stick to a custom simple local search to strictly control budget.
        # "Powell's Method" or "Pattern Search" is good for bound-constrained problems.
        # But with only ~15-20 evals per local search, it might not converge.
        
        # Let's reconsider the DE approach vs Local Search.
        # The prompt says "DIFFERENT strategy".
        # Current: DE with population.
        # New: Intensive Local Search from few seeds.
        
        # Let's implement a simple "Hill Climbing" with adaptive step size.
        # 1. Start at x0.
        # 2. Propose neighbor x' = x + delta * random_dir.
        # 3. If f(x') is better (in terms of weighted sum), accept.
        # 4. Repeat.
        
        # To ensure we find the Pareto front, we should use different weight vectors 
        # for different local searches to cover the front.
        
        n_start = 5
        # Allocate budget:
        # 5 points * 40 evals each = 200.
        
        # Weight vectors for 3 objectives (simplex lattice)
        # We want to cover the front.
        # Weights: (1,0,0), (0,1,0), (0,0,1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1/3, 1/3, 1/3)
        # With 5 starts, let's pick 5 diverse weights.
        weights_list = [
            np.array([1.0, 0.1, 0.1]),
            np.array([0.1, 1.0, 0.1]),
            np.array([0.1, 0.1, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.33, 0.33, 0.34])
        ]
        
        # 1. Initialization
        # Use np.random.seed from the evaluator? 
        # The evaluator calls np.random.seed(seed).
        # We should use np.random directly to respect the seed.
        
        X_starts = []
        for i in range(n_start):
            x = lo + range_ * np.random.random(self.dim)
            X_starts.append(x)
        
        used_evals = 0
        archive_F = []
        
        def add_to_archive(f):
            nonlocal archive_F
            # Check domination
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if dominated:
                return
            # Remove dominated
            new_archive = []
            for af in archive_F:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
            new_archive.append(f)
            archive_F = new_archive

        # Evaluate starts
        F_starts = []
        for x in X_starts:
            f = problem(x)
            used_evals += 1
            F_starts.append(f)
            add_to_archive(f)
            
        if used_evals >= self.budget:
            return F_starts

        # 2. Local Search Phase
        # For each start, run a simple coordinate descent / pattern search.
        # We have (200 - 5) / 5 = 39 evals per local search.
        
        for i in range(n_start):
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = X_starts[i].copy()
            f = F_starts[i].copy()
            w = weights_list[i]
            
            # Objective for local search: Weighted Sum
            obj_val = np.dot(w, f)
            
            # Simple Hill Climbing with random steps, shrinking step size
            step_size = 0.5 # Fraction of range
            
            # Number of steps allowed
            max_steps = (self.budget - used_evals) // n_start # Approx
            
            # Actually, just use all remaining budget for this start? 
            # No, we want to process all starts.
            # Let's allocate fixed budget per start.
            alloc = 35
            
            for step in range(alloc):
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Try moving in a random direction
                direction = np.random.randn(self.dim)
                direction /= (np.linalg.norm(direction) + 1e-8)
                
                x_new = x + step_size * range_ * direction
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used_evals += 1
                obj_new = np.dot(w, f_new)
                
                if obj_new < obj_val:
                    # Accept
                    x = x_new
                    f = f_new
                    obj_val = obj_new
                    add_to_archive(f)
                    # Shrink step size slightly? Or keep?
                    # Let's shrink to refine
                    step_size *= 0.8
                else:
                    # Reject, maybe shrink step size to explore finer
                    step_size *= 0.9
                    if step_size < 0.01:
                        # Stuck, try a random jump?
                        # If stuck, break and move to next start?
                        # Or restart local search with larger step?
                        # Let's just break to save budget for other starts if stuck.
                        # But we have allocated budget.
                        # Let's try a random restart within the local search budget.
                        x = lo + range_ * np.random.random(self.dim)
                        f = problem(x)
                        used_evals += 1
                        obj_val = np.dot(w, f)
                        add_to_archive(f)
                        step_size = 0.5

        # Final return (ignored by evaluator, but must not crash)
        return archive_F
# EVOLVE-BLOCK-END
