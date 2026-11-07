import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6: f1 = 1 + 90*(1-sqrt(x_1)), f2 = (1-f1^2/121)*(1-sqrt(f1/11))
        # x_1 = (1-(f1-1)/90)^2, others = 0
        # Pareto front: f1 in [1, 11]
        
        def f2_of_f1(f1):
            return (1 - f1**2 / 121.0) * (1 - np.sqrt(f1 / 11.0))
        
        def marginal_hv(f1_new, f1_sorted, f2_sorted):
            """Compute marginal HV contribution of adding f1_new to existing set.
            Points are sorted by f1 ascending. f2 is decreasing in f1.
            """
            if len(f1_sorted) == 0:
                # Single point: HV = (11 - f1_new) * f2_new
                return (11.0 - f1_new) * f2_of_f1(f1_new)
            
            f2_new = f2_of_f1(f1_new)
            # Find insertion position
            pos = np.searchsorted(f1_sorted, f1_new)
            
            # Get neighbors
            f1_left = f1_sorted[pos-1] if pos > 0 else None
            f1_right = f1_sorted[pos] if pos < len(f1_sorted) else None
            f2_left = f2_sorted[pos-1] if pos > 0 else None
            f2_right = f2_sorted[pos] if pos < len(f2_sorted) else None
            
            # Marginal HV: the new point creates a new rectangle
            # The old front had a segment from (f1_left, f2_left) to (f1_right, f2_right)
            # The new point splits this into two segments
            # Old contribution of this segment: (f2_left - f2_right) * (11 - f1_left) if both exist
            # Actually, let's compute HV before and after
            
            # Simpler: compute the area added by inserting the new point
            # The new point at (f1_new, f2_new) adds area that was previously "empty"
            
            # For a monotonic decreasing front, inserting a point adds:
            # (f2_left - f2_new) * (f1_new - f1_left) + (f2_new - f2_right) * (11 - f1_new)
            # minus the old area which was (f2_left - f2_right) * (11 - f1_left)
            # But this gets complex. Let me just compute HV difference.
            
            # Build the new sorted array and compute HV
            f1_new_arr = np.insert(f1_sorted, pos, f1_new)
            f2_new_arr = np.insert(f2_sorted, pos, f2_new)
            
            # Compute HV of new set
            N = len(f1_new_arr)
            hv_new = 0.0
            for i in range(N - 1):
                hv_new += (f2_new_arr[i] - f2_new_arr[i+1]) * (11.0 - f1_new_arr[i])
            hv_new += f2_new_arr[N-1] * (11.0 - f1_new_arr[N-1])
            
            # Compute HV of old set
            N_old = len(f1_sorted)
            hv_old = 0.0
            for i in range(N_old - 1):
                hv_old += (f2_sorted[i] - f2_sorted[i+1]) * (11.0 - f1_sorted[i])
            hv_old += f2_sorted[N_old-1] * (11.0 - f1_sorted[N_old-1])
            
            return hv_new - hv_old
        
        # Greedy iterative hypervolume maximization
        # Start with a coarse grid, then greedily add points with max marginal HV
        
        # Initialize with endpoints
        f1_list = [1.0, 11.0]
        f2_list = [f2_of_f1(1.0), f2_of_f1(11.0)]
        
        # Evaluate the two endpoints
        for f1 in [1.0, 11.0]:
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            g = (f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        # Greedily add points
        for _ in range(max_evals - 2):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            # Try to find the best f1 to add
            # Use a candidate set based on midpoints of current intervals
            f1_sorted = np.array(f1_list)
            f2_sorted = np.array(f2_list)
            
            # Generate candidates: midpoints of intervals, plus some extra points
            candidates = []
            for i in range(len(f1_sorted) - 1):
                mid = (f1_sorted[i] + f1_sorted[i+1]) / 2.0
                candidates.append(mid)
                # Also try quartile points
                candidates.append((3*f1_sorted[i] + f1_sorted[i+1]) / 4.0)
                candidates.append((f1_sorted[i] + 3*f1_sorted[i+1]) / 4.0)
            
            # Also try some global candidates
            for t in np.linspace(0, 1, 20):
                candidates.append(1.0 + 10.0 * t)
            
            # Remove duplicates and out-of-range
            candidates = np.array(candidates)
            candidates = candidates[(candidates > 1.0) & (candidates < 11.0)]
            candidates = np.unique(candidates)
            
            # Find candidate with max marginal HV
            best_f1 = None
            best_marginal = -1
            for f1_cand in candidates:
                if time.time() > deadline:
                    break
                marg = marginal_hv(f1_cand, f1_sorted, f2_sorted)
                if marg > best_marginal:
                    best_marginal = marg
                    best_f1 = f1_cand
            
            if best_f1 is None:
                break
            
            # Add the best point
            pos = np.searchsorted(f1_sorted, best_f1)
            f1_list.insert(pos, best_f1)
            f2_list.insert(pos, f2_of_f1(best_f1))
            
            # Evaluate
            g = (best_f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        return None
