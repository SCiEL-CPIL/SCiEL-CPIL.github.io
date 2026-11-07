import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reference point for HV calculation (for pruning/selection)
        # The task says reference point [1705.0, 12.0, 0.3] is used for scoring.
        # We can use this to guide our search.
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initialize archive
        # Archive stores (x, f)
        archive = []
        
        # Helper to check if f_new is dominated by any f in archive
        def is_dominated(f_new, archive_f):
            if len(archive_f) == 0:
                return False
            # f_new is dominated if there exists f in archive_f such that
            # f <= f_new and f != f_new
            # Since we are minimizing:
            # f[i] <= f_new[i] for all i, and f[j] < f_new[j] for at least one j
            # Vectorized check:
            # diff = f_new - archive_f
            # dominated_by = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
            # But we need to be careful with floating point.
            # Let's use standard definition.
            return np.any(np.all(archive_f <= f_new, axis=1) & np.any(archive_f < f_new, axis=1))
            
        # Helper to remove points dominated by f_new
        def remove_dominated(f_new, archive):
            new_archive = []
            for x, f in archive:
                # If f is dominated by f_new, remove it
                # f_new <= f and f_new != f
                if np.all(f_new <= f) and np.any(f_new < f):
                    continue
                new_archive.append((x, f))
            return new_archive

        # Helper to calculate HV contribution of a point to the current front
        # This is expensive to do exactly for every point in the archive repeatedly.
        # For pruning, we can approximate or just keep the non-dominated set.
        # Actually, we don't need to prune aggressively if we just keep the non-dominated set.
        # The size of the non-dominated set for 3 objectives in 5D with 200 points
        # is likely small (< 50). So we can just keep all non-dominated points.
        
        used = 0
        
        # 1. Random Initialization
        init_size = 20
        X_init = lo + (hi - lo) * np.random.rand(init_size, dim)
        for i in range(init_size):
            if time.time() > deadline: break
            x = X_init[i]
            f = np.array(problem(x))
            # Check dominance
            if not is_dominated(f, [f_ for _, f_ in archive]):
                # Remove any points dominated by f
                archive = remove_dominated(f, archive)
                archive.append((x, f))
            used += 1
            
        # 2. Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a parent from the archive
            if not archive:
                # Fallback to random if archive is empty (shouldn't happen)
                x_parent = lo + (hi - lo) * np.random.rand(dim)
            else:
                # Weighted selection: Prefer points with better HV contribution?
                # For simplicity, uniform random from non-dominated set (which is our archive)
                idx = np.random.randint(len(archive))
                x_parent = archive[idx][0]
                
            # Mutate
            # Adaptive sigma: start wide, narrow down?
            # Simple: fixed sigma relative to bounds
            sigma = 0.1 * (hi - lo)
            x_child = x_parent + sigma * np.random.randn(dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = np.array(problem(x_child))
            used += 1
            
            # Update archive
            # 1. Check if f_child is dominated by existing archive
            if is_dominated(f_child, [f_ for _, f_ in archive]):
                continue # Discard, but evaluation is counted
                
            # 2. Remove points dominated by f_child
            archive = remove_dominated(f_child, archive)
            
            # 3. Add f_child
            archive.append((x_child, f_child))
            
            # Optional: Pruning if archive gets too large
            # With 200 points, the non-dominated front in 3D is usually small.
            # But if it grows, we can prune.
            if len(archive) > 100:
                # Keep the 100 points with the best HV contribution
                # This requires HV calculation.
                # Let's skip complex pruning for now and rely on the fact that
                # dominated points are removed. The non-dominated set size is bounded.
                pass

        return archive
