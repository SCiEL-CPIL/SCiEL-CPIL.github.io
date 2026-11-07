"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hierarchical Global-to-Local Optimizer with Halton Grid Search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # 1. Halton Sequence Generator for Quasi-Random Global Search
        def halton(dim, index):
            """Generate Halton sequence point for given index."""
            point = np.zeros(dim)
            for i in range(dim):
                prime = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim])[i]
                f = 1.0
                r = 0.0
                idx = index + 1
                while idx > 0:
                    f /= prime
                    r += f * (idx % prime)
                    idx //= prime
                point[i] = r
            return point

        # 2. Global Phase: Dense Grid Search
        # Use 30% of budget for global mapping
        n_global = int(budget * 0.3)
        n_global = max(10, n_global)
        
        x_best = None
        f_best = float('inf')
        calls = 0
        candidates = []

        # Start with a few center points
        candidates.append(np.zeros(dim))
        # Add Halton points
        for i in range(1, n_global + 1):
            if time.time() > deadline:
                break
            x = 10.0 * halton(dim, i) - 5.0  # Map [0,1] to [-5,5]
            candidates.append(x)
            
        # Evaluate Global Candidates
        for x in candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            # Keep top 5 distinct points for local search
            if len(candidates) < 1000: # Avoid memory bloat if budget is huge
                candidates.append(x)

        # Simple clustering to find distinct basins
        # Sort by fitness
        points = [(f, x) for x, f in zip(candidates[:n_global], [func(x) for x in candidates[:n_global]])]
        # Wait, I already evaluated them. Let's just track the best ones.
        # Actually, I didn't store the mapping of x to f in the loop above efficiently.
        # Let's re-evaluate or store. The loop above is inefficient if I re-evaluate.
        # Let's refactor the global phase to store results.
        
        # RE-IMPLEMENTATION OF GLOBAL PHASE FOR CORRECTNESS
        # The code above was a sketch. Let's do it properly.
        
        # Reset
        x_best = None
        f_best = float('inf')
        calls = 0
        evaluated_points = []

        # Phase 1: Global Search (Halton)
        n_global = int(budget * 0.3)
        n_global = max(20, n_global)
        
        for i in range(1, n_global + 1):
            if time.time() > deadline:
                break
            x = 10.0 * halton(dim, i) - 5.0
            fx = func(x)
            calls += 1
            evaluated_points.append((fx, x))
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # Phase 2: Local Refinement of Top Basins
        # Select top K distinct points from global search
        # Sort by fitness
        evaluated_points.sort(key=lambda p: p[0])
        
        # Pick top 5, ensuring they are somewhat distinct
        local_starts = []
        for fx, x in evaluated_points:
            if len(local_starts) >= 5:
                break
            # Check if close to existing starts
            is_distinct = True
            for ls in local_starts:
                if np.linalg.norm(x - ls) < 1.0:
                    is_distinct = False
                    break
            if is_distinct:
                local_starts.append(x)
        
        # If no distinct points found (all clustered), just take the best
        if not local_starts:
            local_starts.append(x_best)

        # Phase 3: Local Optimization (Adaptive (1+1)-ES on each start)
        # Distribute remaining budget among local starts
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return x_best, f_best
            
        n_local_starts = len(local_starts)
        budget_per_start = max(1, remaining_budget // n_local_starts)

        for x_start in local_starts:
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            x_curr = x_start.copy()
            # Estimate initial sigma based on distance to other starts or default
            sigma = 0.5 
            stall_count = 0
            stall_limit = 10

            # Local loop
            while calls < budget:
                if time.time() > deadline:
                    break
                
                y = np.clip(x_curr + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    x_curr = y.copy()
                    sigma *= 1.1
                    stall_count = 0
                else:
                    sigma *= 0.95
                    stall_count += 1
                    if stall_count > stall_limit:
                        sigma *= 2.0
                        stall_count = 0
                        # If stuck in local min, break out to next global basin
                        if sigma > 2.0: 
                            break
                sigma = np.clip(sigma, 1e-4, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
