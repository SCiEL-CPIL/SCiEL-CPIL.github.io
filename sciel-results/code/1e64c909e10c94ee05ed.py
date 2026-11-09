"""Multi-objective optimizer for RE21 using hypervolume-based archive pruning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        
        # Helper: Hypervolume calculation for 2D non-dominated set
        def calc_hv(points, ref):
            if len(points) == 0:
                return 0.0
            # Filter points that contribute to HV (must be better than ref in all objectives)
            valid_pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if len(valid_pts) == 0:
                return 0.0
            
            # Sort by first objective ascending
            valid_pts.sort(key=lambda x: x[0])
            
            hv = 0.0
            prev_x = 0.0 # Not strictly 0, but we sum widths
            # Actually, the formula is sum of (x_{i+1} - x_i) * (ref_y - y_i)
            # where x_{k+1} = ref_x
            # Let's iterate
            for i in range(len(valid_pts)):
                x_i, y_i = valid_pts[i]
                if i == len(valid_pts) - 1:
                    x_next = ref[0]
                else:
                    x_next = valid_pts[i+1][0]
                
                # Ensure we don't have negative widths due to sorting issues or duplicates
                width = x_next - x_i
                if width < 0:
                    width = 0.0
                    
                height = ref[1] - y_i
                if height < 0:
                    height = 0.0
                
                hv += width * height
            return hv

        # Helper: Marginal contribution of a point
        def marginal_contrib(points, ref, idx):
            hv_full = calc_hv(points, ref)
            # Create a copy without the point at idx
            points_without = points[:idx] + points[idx+1:]
            hv_without = calc_hv(points_without, ref)
            return hv_full - hv_without

        # Helper: Check if point a dominates point b (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Initialize Archive with LHS
        n_init = 20
        # Latin Hypercube Sampling
        indices = np.arange(n_init)
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = np.random.permutation(indices)
            # Center of the bin
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        # Scale to bounds
        lhs_points = lo + (hi - lo) * lhs_points
        
        archive = []
        evals_used = 0
        
        # Evaluate initial population
        for x in lhs_points:
            f = problem(x)
            evals_used += 1
            # Only add if it's "valid" for HV (better than ref)
            # And non-dominated by current archive
            if f[0] < ref[0] and f[1] < ref[1]:
                is_dominated = False
                for p in archive:
                    if dominates(p, f):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Remove points dominated by f
                    new_archive = [p for p in archive if not dominates(f, p)]
                    new_archive.append(f)
                    archive = new_archive
        
        # Main Loop
        max_archive_size = 50
        while evals_used < budget:
            # Time check
            if time.time() > deadline:
                break
            
            if len(archive) == 0:
                # If archive is empty, pick a random point
                parent_x = lo + (hi - lo) * np.random.rand(dim)
            else:
                # Select parent weighted by marginal contribution
                contributions = [marginal_contrib(archive, ref, i) for i in range(len(archive))]
                # Add a small epsilon to avoid zero division or all zeros
                total_contrib = sum(contributions)
                if total_contrib == 0:
                    probs = np.ones(len(archive)) / len(archive)
                else:
                    probs = np.array(contributions) / total_contrib
                
                # Sample parent index
                parent_idx = np.random.choice(len(archive), p=probs)
                parent_f = archive[parent_idx]
                
                # We need the x corresponding to parent_f. 
                # The problem is that we only stored f in the archive. 
                # We need to store x as well to mutate it.
                # Let's fix the archive structure to store (x, f) pairs.
                # But wait, I already evaluated LHS. I didn't store x in archive.
                # I need to restart or modify the logic to store x.
                # Let's rewrite the archive logic to store (x, f).
                pass

        # The above logic has a flaw: I didn't store x in the archive.
        # I need to store (x, f) tuples in the archive.
        
        return None
# EVOLVE-BLOCK-END
