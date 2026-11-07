"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using LHS initialization and greedy hypervolume-guided search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([1.1, 1.3, 1.1]) # Based on task description

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Latin Hypercube Sampling for initial population
        n_init = 20
        # LHS: divide each dimension into n_init intervals, pick one point per interval
        # For simplicity and speed, we can just use random if LHS is complex, 
        # but LHS is better for coverage.
        # Simple LHS implementation:
        # Create a grid of indices 0..n_init-1, shuffle for each dimension
        lhs_indices = np.array([np.random.permutation(n_init) for _ in range(self.dim)]).T
        # Map to [0, 1]
        u = (lhs_indices + np.random.random((n_init, self.dim))) / n_init
        X_init = lo + (hi - lo) * u
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        evals = n_init
        
        # Archive: list of [x, f]
        archive = list(zip(X_init, F_init))
        
        # Helper: Check if point a dominates point b (a is better in all, strictly better in at least one)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
            
        # Helper: Calculate Hypervolume contribution of a point given the set of non-dominated points
        # For 3 objectives, HV contribution is the volume of the box defined by the point and the reference point,
        # minus the volume already covered by other non-dominated points.
        # This is complex to calculate exactly for arbitrary sets.
        # Approximation: We can just keep all non-dominated points if the number is small.
        # If the archive gets too large, we prune based on a simple metric.
        
        max_archive_size = 50
        
        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select parent
            # Get non-dominated points from archive
            non_dom = []
            for x, f in archive:
                is_dom = False
                for x2, f2 in archive:
                    if not np.array_equal(f, f2): # Avoid self
                        if dominates(f2, f):
                            is_dom = True
                            break
                if not is_dom:
                    non_dom.append((x, f))
            
            if not non_dom:
                # Fallback if all dominated (should not happen with pruning)
                parent_x, parent_f = archive[np.random.randint(len(archive))]
            else:
                # Pick a random non-dominated point
                parent_x, parent_f = non_dom[np.random.randint(len(non_dom))]
                
            # Mutate
            # Use a step size that shrinks slightly or is fixed
            # Fixed step size relative to bounds
            step = 0.1
            noise = np.random.standard_normal(self.dim) * step
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = np.array(problem(child_x))
            evals += 1
            
            # Update Archive
            # 1. Remove points dominated by child
            new_archive = []
            for x, f in archive:
                if not dominates(child_f, f):
                    new_archive.append((x, f))
            
            # 2. Check if child is dominated by any remaining point
            is_child_dom = False
            for x, f in new_archive:
                if dominates(f, child_f):
                    is_child_dom = True
                    break
            
            if not is_child_dom:
                new_archive.append((child_x, child_f))
                
            # 3. Prune if archive is too large
            if len(new_archive) > max_archive_size:
                # Keep points with highest "spread" or just random?
                # A better metric: Keep points that contribute most to HV.
                # Since exact HV contribution is hard, we can use a simple crowding distance or just random removal.
                # Let's try to remove the point that is most "central" (worst for diversity).
                # Or simply, if we have many non-dominated, the front is well-sampled.
                # Let's just keep the first max_archive_size points? No, that's biased.
                # Let's remove a random dominated point if any, else random non-dominated.
                # Actually, if we have >50 non-dominated points, the front is dense.
                # We can just keep the non-dominated ones and if that's >50, remove random.
                dom_points = []
                non_dom_points = []
                for x, f in new_archive:
                    is_d = False
                    for x2, f2 in new_archive:
                        if not np.array_equal(f, f2):
                            if dominates(f2, f):
                                is_d = True
                                break
                    if is_d:
                        dom_points.append((x, f))
                    else:
                        non_dom_points.append((x, f))
                
                if len(dom_points) > 0:
                    # Remove a random dominated point
                    idx = np.random.randint(len(dom_points))
                    point_to_remove = dom_points[idx]
                    new_archive.remove(point_to_remove)
                else:
                    # All non-dominated. Remove one with lowest "value" (sum of objectives?)
                    # Or just random.
                    # Let's remove the one with the highest sum of objectives (worst average)
                    sums = [np.sum(f) for _, f in new_archive]
                    idx = np.argmax(sums)
                    new_archive.pop(idx)
                    
            archive = new_archive

        return archive
# EVOLVE-BLOCK-END
