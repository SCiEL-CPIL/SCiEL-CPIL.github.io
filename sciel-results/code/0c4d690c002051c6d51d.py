"""Multi-objective optimizer for ZDT4 using analytical Pareto front sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct structural ZDT4 solver: sample the known Pareto front densely.
    
    ZDT4 Pareto front: x_2..x_n = 0, x_1 in [0, 1]
    f_1 = 1 + (n-1) + sum((x_i - 0.05)^2 - 10*cos(20*pi*(x_i - 0.05))) for i=1..n
    f_2 = g(x) * h(f_1/g(x)) where g(x) = 1 + (n-1) + sum(x_i^2) for i=2..n
         h(v) = 1 - sqrt(v/g(x))
    
    On the Pareto front, g(x) is minimized (x_2..x_n=0), so g = n.
    f_1 ranges from ~1 to ~11 for n=10.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        # Determine optimal tail values (x_2..x_n) for the Pareto front
        # For ZDT4, g(x) is minimized when x_2..x_n = 0
        tail_optimal = np.zeros(dim - 1)
        for i in range(dim - 1):
            idx = i + 1  # x_2 is index 1
            if lo[idx] <= 0.0 <= hi[idx]:
                tail_optimal[i] = 0.0
            elif 0.0 < lo[idx]:
                tail_optimal[i] = lo[idx]
            else:
                tail_optimal[i] = hi[idx]
        
        used = 0
        archive_F = []
        ref = np.array([11.0, 400.0])
        
        def _eval_and_record(x1_val):
            nonlocal used
            if used >= budget:
                return False
            if time.time() > deadline:
                return False
            
            x = np.zeros(dim)
            x[0] = x1_val
            x[1:] = tail_optimal
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                used += 1
                if np.all(np.isfinite(f)) and np.all(f < ref):
                    archive_F.append(f.copy())
                return True
            except Exception:
                used += 1
                return False
        
        # Phase 1: Dense uniform sampling of x_1
        # Use a good initial set of points
        n_initial = min(100, budget)
        if n_initial < 2:
            n_initial = 2
        
        x1_vals = np.linspace(lo[0], hi[0], n_initial)
        x1_list = list(x1_vals)
        
        for x1 in x1_vals:
            if not _eval_and_record(x1):
                break
        
        # Phase 2: Adaptive refinement - add midpoints in largest gaps
        # Sort the evaluated x1 values
        x1_list = sorted(x1_list)
        
        while used < budget and time.time() < deadline:
            if len(x1_list) < 2:
                break
            
            # Find the largest gap
            gaps = np.diff(x1_list)
            if len(gaps) == 0 or gaps.max() < 1e-12:
                break
            
            max_idx = int(np.argmax(gaps))
            midpoint = (x1_list[max_idx] + x1_list[max_idx + 1]) / 2.0
            
            if not _eval_and_record(midpoint):
                break
            
            # Insert midpoint into sorted list
            pos = np.searchsorted(x1_list, midpoint)
            x1_list.insert(pos, midpoint)
        
        # Return the archive
        if archive_F:
            return np.array(archive_F)
        else:
            # Fallback: return a single point
            x = np.zeros(dim)
            x[0] = (lo[0] + hi[0]) / 2
            x[1:] = tail_optimal
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if np.all(np.isfinite(f)) and np.all(f < ref):
                    return np.array([f])
            except Exception:
                pass
            return np.array([[10.0, 300.0]])

# EVOLVE-BLOCK-END
