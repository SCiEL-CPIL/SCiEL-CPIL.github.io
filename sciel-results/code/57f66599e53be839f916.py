"""
Coarse-to-fine grid search optimizer for BBOB f3 (Rastrigin).
Uses deterministic systematic partitioning to guarantee global coverage,
then refines the best region with local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 320s limit, keep 10s margin for safety
        deadline = start_time + 320

        n = self.dim
        budget = self.budget

        # State
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse-to-fine grid search
        # Start with a coarse grid over [-5, 5]^n
        # The grid size grows adaptively as we refine the best region
        
        # Initial coarse grid: 3 points per dimension = 3^n points
        # For n=5, that's 243 points, which fits in our budget
        # Then refine the best cell with a finer grid
        
        # We'll use a recursive subdivision approach:
        # 1. Evaluate a coarse grid
        # 2. Find the best cell
        # 3. Subdivide the best cell (and possibly neighboring cells)
        # 4. Repeat until budget is exhausted or convergence
        
        # Define the search space
        lb = -5.0
        ub = 5.0
        
        # Initial grid: 3 points per dimension
        grid_size = 3
        # Number of cells per dimension = grid_size - 1 = 2
        # We evaluate the center of each cell plus the boundaries? 
        # Actually, for Rastrigin, the minimum is at the center of a cell,
        # so evaluating cell centers is good.
        
        # Let's use a different approach: evaluate points on a grid
        # where we have grid_size points per dimension, including boundaries
        # Then the best point tells us which region to refine
        
        # Phase 1: Coarse grid
        # 3 points per dim: -5, 0, 5 -> 3^5 = 243 evaluations
        # This is a lot but fits in 500 budget
        
        # Actually, let's be smarter. Use 2 points per dim initially:
        # -5, 5 -> 2^5 = 32 evaluations
        # Then refine the best half-space
        
        # Better: use a space-filling approach. Start with center,
        # then expand outward.
        
        # Let's do: 
        # 1. Evaluate center (1 call)
        # 2. Evaluate 2n points at +/-1 along each axis (10 calls for n=5)
        # 3. Evaluate 2n points at +/-2 along each axis (10 calls)
        # 4. Evaluate 2n points at +/-3 along each axis (10 calls)
        # 5. Evaluate 2n points at +/-4 along each axis (10 calls)
        # 6. Evaluate corners (2^n = 32 calls)
        # Total: 1 + 40 + 32 = 73 calls for broad exploration
        
        # Then use remaining budget for local refinement
        
        # Phase 1a: Center and axis-aligned exploration
        points = [np.zeros(n)]
        
        # Add axis-aligned points at various distances
        for dist in [1.0, 2.0, 3.0, 4.0]:
            for i in range(n):
                for sign in [1, -1]:
                    x = np.zeros(n)
                    x[i] = sign * dist
                    points.append(x)
        
        # Add corners of the box
        for mask in range(2**n):
            x = np.zeros(n)
            for i in range(n):
                if mask & (1 << i):
                    x[i] = 5.0
                else:
                    x[i] = -5.0
            points.append(x)
        
        # Deduplicate and evaluate
        seen = set()
        for x in points:
            x = np.clip(x, -5, 5)
            key = tuple(np.round(x, 6))
            if key in seen:
                continue
            seen.add(key)
            if check_deadline():
                break
            f = func(x)
            update_best(x, f)
        
        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f
        
        # Phase 1b: Coarse grid around the best region
        # Determine the best region from Phase 1a
        # Find the bounding box of points that are close to the best
        
        # Actually, let's do a proper coarse-to-fine grid search:
        # Define the current search region
        region_lb = np.full(n, -5.0)
        region_ub = np.full(n, 5.0)
        
        # Coarse grid: 3 points per dimension
        n_coarse = 3
        for _ in range(3):  # 3 levels of refinement
            if check_deadline():
                break
            
            # Generate grid points
            grid_points = []
            for i in range(n_coarse):
                t = i / (n_coarse - 1) if n_coarse > 1 else 0.5
                coord = region_lb + t * (region_ub - region_lb)
                grid_points.append(coord)
            
            # Cartesian product
            from itertools import product
            all_points = list(product(*grid_points))
            
            # Evaluate all grid points
            best_local_x = None
            best_local_f = float('inf')
            for point in all_points:
                if check_deadline():
                    break
                x = np.array(point)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                if f < best_local_f:
                    best_local_f = f
                    best_local_x = x.copy()
            
            if best_local_x is None:
                break
            
            # Refine: focus on the region around the best point
            # Shrink the region to half the size around the best point
            center = best_local_x
            half_width = (region_ub - region_lb) / 2.0
            region_lb = center - half_width / 2.0
            region_ub = center + half_width / 2.0
            region_lb = np.clip(region_lb, -5, 5)
            region_ub = np.clip(region_ub, -5, 5)
            
            # Increase grid resolution
            n_coarse = min(n_coarse + 1, 6)
        
        # Phase 2: Local refinement using a simple (1+1)-ES
        # Starting from the best point found by grid search
        
        if check_deadline() or calls_made >= budget:
            return best_x, best_f
        
        # Local search with adaptive step size
        x = best_x.copy()
        f = best_f
        step_size = 0.5
        rng = np.random.default_rng(42)
        
        no_improve = 0
        max_no_improve = 20
        
        while not check_deadline() and calls_made < budget and no_improve < max_no_improve:
            # Try a random perturbation
            x_try = x + step_size * rng.standard_normal(n)
            x_try = np.clip(x_try, -5, 5)
            f_try = func(x_try)
            calls_made += 1
            
            if f_try < f:
                x = x_try
                f = f_try
                best_f = f
                best_x = x.copy()
                no_improve = 0
                # Slightly increase step size on success
                step_size = min(step_size * 1.2, 5.0)
            else:
                no_improve += 1
                # Decrease step size on failure
                step_size *= 0.8
                
            if step_size < 1e-6:
                break
        
        # Phase 3: If we have budget left, do another coarse-to-fine pass
        # from a different starting region to avoid local minima
        
        remaining = budget - calls_made
        if remaining > 50 and not check_deadline():
            # Try a few random restarts with local search
            for _ in range(min(5, remaining // 30)):
                if check_deadline() or calls_made >= budget:
                    break
                
                x_restart = rng.uniform(-4.5, 4.5, n)
                f_restart = func(x_restart)
                calls_made += 1
                
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                
                # Quick local search from this point
                x_local = x_restart.copy()
                f_local = f_restart
                step = 0.5
                for _ in range(10):
                    if check_deadline() or calls_made >= budget:
                        break
                    x_try = x_local + step * rng.standard_normal(n)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        if f_local < best_f:
                            best_f = f_local
                            best_x = x_local.copy()
                        step = min(step * 1.1, 3.0)
                    else:
                        step *= 0.85
                    if step < 1e-5:
                        break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
