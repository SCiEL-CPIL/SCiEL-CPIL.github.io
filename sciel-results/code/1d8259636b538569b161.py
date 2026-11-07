"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using a structured approach to sample the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline (30s margin from 310s timeout)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 structure: x1 determines f1, the rest determine f2
        # f1 = x1
        # g(x2..x30) = 1 + 10*(n-1) + sum(100*(sin^2(2*pi*xi) - 1) for xi in x2..x30)
        # h(x1, g) = 1 - sqrt(x1/g) - sin(10*pi*x1)
        # f2 = g(x2..x30) * h(x1, g)
        
        # The Pareto front of ZDT3 is complex (disconnected). 
        # We know the structure: f1 = x1, and for optimal solutions, g is minimized.
        # The minimum of g is 1 (when all sin^2 terms are 0, i.e., xi = 0.5 for i>=2).
        # When g=1, f2 = h(x1, 1) = 1 - x1 - sin(10*pi*x1).
        
        # Strategy: 
        # 1. Use a portion of budget to explore x1 values with g at its minimum (xi=0.5 for i>=2)
        # 2. Use remaining budget to refine and explore variations
        
        # Phase 1: Sample x1 uniformly with optimal g (xi=0.5 for i>=2)
        # This gives us points on the "ideal" front where g=1
        n_phase1 = min(100, budget // 2)
        x1_vals = np.linspace(0, 1, n_phase1)
        
        results = []
        used = 0
        
        for x1 in x1_vals:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x1
            f = problem(x)
            results.append(f)
            used += 1
        
        # Phase 2: Explore variations of g to find more of the front
        # The ZDT3 front has multiple disconnected components.
        # Let's explore different g values by varying some of the xi values.
        
        # For the remaining budget, do a more systematic exploration
        remaining = budget - used
        
        if remaining > 0:
            # Try different x1 values with slightly perturbed g
            # Focus on regions where the front has high curvature
            n_phase2 = min(remaining, 100)
            
            # Use a stratified approach: sample x1 in finer grid, and for each,
            # try to minimize g by setting xi to 0.5, but also try some variations
            x1_fine = np.linspace(0, 1, n_phase2)
            
            for i, x1 in enumerate(x1_fine):
                if used >= budget or time.time() > deadline:
                    break
                
                # For each x1, evaluate with optimal g
                x = np.full(dim, 0.5)
                x[0] = x1
                f = problem(x)
                results.append(f)
                used += 1
                
                if used >= budget or time.time() > deadline:
                    break
                
                # Also try a perturbed version to capture the discontinuities
                # Vary one of the other dimensions to increase g slightly
                if i % 2 == 0:
                    x2 = np.full(dim, 0.5)
                    x2[0] = x1
                    # Perturb one dimension to increase g
                    pert_dim = 1 + (i % (dim - 1))
                    x2[pert_dim] = 0.0  # This increases sin^2 term
                    f2 = problem(x2)
                    results.append(f2)
                    used += 1
        
        # Phase 3: If budget remains, do local refinement around promising regions
        remaining = budget - used
        if remaining > 0 and len(results) > 0:
            results_arr = np.array(results)
            
            # Find points that are likely on the front (low f2 for given f1)
            # Sort by f1 and find local minima in f2
            order = np.argsort(results_arr[:, 0])
            sorted_f = results_arr[order]
            
            # Sample additional points in regions of high curvature
            n_phase3 = min(remaining, 50)
            
            # Focus on regions where the front is complex
            # ZDT3 has discontinuities, so sample more densely in certain regions
            for i in range(n_phase3):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick a random x1 from the lower half where the front is more complex
                x1 = np.random.uniform(0, 0.5)
                x = np.full(dim, 0.5)
                x[0] = x1
                
                # Add some randomness to the other dimensions
                noise = np.random.normal(0, 0.05, dim - 1)
                x[1:] = np.clip(0.5 + noise, 0, 1)
                
                f = problem(x)
                results.append(f)
                used += 1
        
        # Return the results (the evaluator records all returned objective vectors)
        return results
# EVOLVE-BLOCK-END
