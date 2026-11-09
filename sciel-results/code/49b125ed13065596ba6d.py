"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using batched multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for normalization (based on task description)
        ref = np.array([1705.0, 12.0, 0.3])
        
        # We will use a fixed number of local search runs.
        # Each run consumes roughly (dim + 1) * iterations.
        # Let's aim for ~10-15 runs to stay well within budget and time.
        # If local search is fast, we can do more.
        # Let's start with 20 runs.
        n_runs = 20
        
        # Ensure we don't exceed budget. 
        # A typical Nelder-Mead run on 5 vars takes ~50-100 evals to converge.
        # 20 runs * 100 evals = 2000 evals. This is way over budget 200.
        # So we must be very careful. 
        # Budget is 200. 
        # If we do 10 runs, we have 20 evals per run.
        # Nelder-Mead with 5 vars needs at least 6 points (dim+1). 
        # So 20 evals is ~3-4 iterations. This is very short.
        
        # Alternative: Use fewer, more targeted runs? 
        # Or use a simpler local search? 
        # The seed program used a single population of 20 and 180 mutations.
        # Let's try 5 local search runs, each with ~40 budget.
        # 40 evals / 6 points per simplex = ~6 iterations. Still short.
        
        # Maybe the "local search" should be a coarse grid or random restarts with very few steps?
        # Or perhaps we should use the full budget for a single, very well-tuned local search? 
        # No, multi-objective needs diversity.
        
        # Let's stick to the strategy: Multi-start local search.
        # We will distribute the budget among N starts.
        # Let N = 10. Budget per start = 20.
        # 20 evals is enough to initialize a simplex (6 evals) and do 2-3 steps.
        
        # Actually, looking at the seed program, it did 200 evals total.
        # Let's try 10 restarts.
        
        rng = np.random.default_rng(0)
        
        # Generate initial points for the local searches
        # Use a mix of random and structured points (corners, midpoints)
        init_points = []
        
        # Corners
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = hi[j]
                else:
                    p[j] = lo[j]
            init_points.append(p)
            
        # Midpoints
        mid = (lo + hi) / 2
        init_points.append(mid)
        
        # Random points to fill up to n_runs
        while len(init_points) < n_runs:
            init_points.append(lo + (hi - lo) * rng.random(dim))
            
        # Shuffle to ensure randomness in order
        idx = rng.permutation(len(init_points))
        init_points = [init_points[i] for i in idx]
        
        archive = []
        used_budget = 0
        
        def is_dominated(f, archive_f):
            # Check if f is dominated by any point in archive_f
            if len(archive_f) == 0:
                return False
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
            
        def update_archive(f, x):
            nonlocal archive
            # Remove points dominated by f
            new_archive = []
            for i, (af, ax) in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((af, ax))
            # Add f if not dominated
            if not is_dominated(f, [a[0] for a in new_archive]):
                new_archive.append((f, x))
            archive = new_archive

        for i in range(n_runs):
            if used_budget >= budget:
                break
            if time.time() > deadline:
                break
                
            x0 = init_points[i]
            f0 = problem(x0)
            used_budget += 1
            update_archive(f0, x0)
            
            # Perform a simple local search (Nelder-Mead style)
            # We will use a small budget for this local search
            local_budget = min(20, budget - used_budget)
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [f0.copy()]
            
            # Add dim more points to simplex
            for j in range(dim):
                if used_budget >= budget or time.time() > deadline:
                    break
                xj = x0.copy()
                # Perturb in dimension j
                step = 0.1 * (hi[j] - lo[j])
                xj[j] += step
                xj = np.clip(xj, lo, hi)
                fj = problem(xj)
                used_budget += 1
                simplex.append(xj)
                f_simplex.append(fj)
                update_archive(fj, xj)
                
            # Simplex optimization loop
            # We'll do a few iterations of Nelder-Mead
            # Objective: sum of normalized objectives
            def obj(x):
                fx = problem(x)
                # Normalize by reference point to make scale comparable
                # If fx > ref, it's bad. We want to minimize.
                # Simple sum of raw objectives might be dominated by large values.
                # Let's use the raw sum for now, or normalized.
                # ref is [1705, 12, 0.3].
                # Let's try sum of (fx / ref) to normalize scale.
                # But if fx > ref, ratio > 1.
                # We want to minimize.
                return np.sum(fx / ref)
                
            # Note: We already evaluated x0 and the initial simplex points.
            # We need to be careful not to re-evaluate unnecessarily, but for simplicity 
            # and given the small budget, we might just let Nelder-Mead re-evaluate 
            # or manage the cache. Given the tight budget (200 total), re-evaluation is costly.
            # Let's implement a simple step-based local search instead of full NM to save code complexity 
            # and ensure we stay within budget strictly.
            
            # Simple Hill Climbing / Coordinate Descent
            current_x = x0
            current_f = f0
            current_obj = obj(current_x)
            
            # We have used 1 + dim evals so far for this start.
            # Remaining for this start: local_budget - dim
            # Let's do coordinate ascent/descent steps.
            step_size = 0.1
            max_steps = 5 # Keep it small
            
            for step in range(max_steps):
                if used_budget >= budget or time.time() > deadline:
                    break
                    
                best_x = current_x
                best_f = current_f
                best_obj = current_obj
                
                for j in range(dim):
                    if used_budget >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    xp = current_x.copy()
                    xp[j] += step_size * (hi[j] - lo[j])
                    xp = np.clip(xp, lo, hi)
                    fp = problem(xp)
                    used_budget += 1
                    update_archive(fp, xp)
                    op = obj(xp)
                    if op < best_obj:
                        best_obj = op
                        best_x = xp
                        best_f = fp
                        
                    # Try negative step
                    xn = current_x.copy()
                    xn[j] -= step_size * (hi[j] - lo[j])
                    xn = np.clip(xn, lo, hi)
                    fn = problem(xn)
                    used_budget += 1
                    update_archive(fn, xn)
                    on = obj(xn)
                    if on < best_obj:
                        best_obj = on
                        best_x = xn
                        best_f = fn
                        
                if np.allclose(best_x, current_x):
                    break
                current_x = best_x
                current_f = best_f
                current_obj = best_obj
                # Shrink step size if no improvement? 
                # For simplicity, keep it fixed or shrink slightly.
                step_size *= 0.5

        return archive
# EVOLVE-BLOCK-END
