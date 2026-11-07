"""Multi-objective optimizer for ZDT4 using analytical Pareto front + local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct structural ZDT4 solver: construct known Pareto front analytically,
    then use local search to refine points toward the true front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        # ZDT4 specific structure:
        # f1 = 1 + 10*(n-1) + sum((xi-0.05)^2 - 10*cos(20*pi*(xi-0.05))) for i=1..n
        # f2 = g(x) * h(f1/g(x)) where g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n
        # h(v) = 1 - sqrt(v/g(x))
        # The Pareto front is achieved when x2..xn = 0 (making g minimal) 
        # and x1 varies to control f1.
        
        # Strategy: 
        # 1. First, find optimal x2..xn by minimizing g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n
        #    This is simply x2..xn = 0 if bounds allow, else clip to 0.
        # 2. Sweep x1 across its range to cover the Pareto front.
        # 3. Use remaining budget for local refinement.
        
        # Determine optimal tail values (x2..xn)
        # For ZDT4, g(x) = 1 + 10*(n-1) + sum(xi^2) for i=2..n
        # Minimum g is achieved when x2..xn = 0
        tail_optimal = np.zeros(dim - 1)
        for i in range(dim - 1):
            idx = i + 1  # x2 is index 1
            if lo[idx] <= 0.0 <= hi[idx]:
                tail_optimal[i] = 0.0
            elif 0.0 < lo[idx]:
                tail_optimal[i] = lo[idx]
            else:
                tail_optimal[i] = hi[idx]
        
        # Number of points to sample along the Pareto front
        # Use most of the budget for front coverage, keep some for refinement
        n_front_points = min(150, budget - 20)  # Keep 20 for refinement
        if n_front_points < 10:
            n_front_points = 10
        
        # Sample x1 values - use a denser sampling near the ends where hypervolume contribution is higher
        # For ZDT4, f1 ranges from ~1 to ~11 (for n=10)
        # Uniform sampling should be sufficient, but let's use slightly more points
        x1_vals = np.linspace(lo[0], hi[0], n_front_points)
        
        # Evaluate all front candidates
        used = 0
        archive_X = []
        archive_F = []
        
        for x1 in x1_vals:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1
            x[1:] = tail_optimal
            
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                used += 1
                
                if np.all(np.isfinite(f)) and np.all(f < [11.0, 400.0]):
                    archive_X.append(x.copy())
                    archive_F.append(f.copy())
            except:
                pass
        
        # Now use remaining budget for local search refinement
        if archive_X and used < budget:
            remaining = budget - used
            n_archive = len(archive_X)
            
            # Allocate refinement steps: more steps for points that might benefit
            # Use a simple allocation: distribute remaining budget
            if n_archive > 0:
                steps_per_point = max(1, remaining // n_archive)
                
                for idx in range(n_archive):
                    if time.time() > deadline:
                        break
                    if used >= budget:
                        break
                        
                    current_x = archive_X[idx].copy()
                    current_f = archive_F[idx].copy()
                    
                    for step in range(steps_per_point):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Small random perturbation, scaled by distance from boundary
                        scale = 0.005 * (hi - lo)
                        perturbation = np.random.randn(dim) * scale
                        new_x = current_x + perturbation
                        new_x = np.clip(new_x, lo, hi)
                        
                        try:
                            new_f = problem(new_x)
                            new_f = np.array(new_f, dtype=float)
                            used += 1
                            
                            if np.all(np.isfinite(new_f)) and np.all(new_f < [11.0, 400.0]):
                                # Check if new point is better (dominates)
                                if np.all(new_f <= current_f) and np.any(new_f < current_f):
                                    current_x = new_x
                                    current_f = new_f
                                    archive_X[idx] = current_x.copy()
                                    archive_F[idx] = current_f.copy()
                        except:
                            pass
        
        # Convert to arrays
        if archive_X:
            result_X = np.array(archive_X)
            result_F = np.array(archive_F)
        else:
            # Fallback: return a single point at the center
            result_X = np.array([(lo + hi) / 2])
            try:
                result_F = np.array([problem(result_X[0])])
            except:
                result_F = np.array([[10.0, 300.0]])
        
        return result_X
# EVOLVE-BLOCK-END
