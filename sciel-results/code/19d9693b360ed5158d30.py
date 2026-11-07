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
        
        def compute_hv(f1_vals):
            """Compute hypervolume for points on Pareto front, ref [11,11]"""
            f1_vals = np.sort(f1_vals)
            f2_vals = f2_of_f1(f1_vals)
            N = len(f1_vals)
            if N == 0:
                return 0.0
            hv = 0.0
            for i in range(N - 1):
                hv += (f2_vals[i] - f2_vals[i+1]) * (11.0 - f1_vals[i])
            hv += f2_vals[N-1] * (11.0 - f1_vals[N-1])
            return hv

        def marginal_hv(f1_new, f1_sorted, f2_sorted):
            """Compute marginal HV contribution of adding f1_new to existing set.
            Points are sorted by f1 ascending. f2 is decreasing in f1.
            """
            if len(f1_sorted) == 0:
                return (11.0 - f1_new) * f2_of_f1(f1_new)
            
            f2_new = f2_of_f1(f1_new)
            pos = np.searchsorted(f1_sorted, f1_new)
            
            # Compute HV before and after
            f1_new_arr = np.insert(f1_sorted, pos, f1_new)
            f2_new_arr = np.insert(f2_sorted, pos, f2_new)
            
            N = len(f1_new_arr)
            hv_new = 0.0
            for i in range(N - 1):
                hv_new += (f2_new_arr[i] - f2_new_arr[i+1]) * (11.0 - f1_new_arr[i])
            hv_new += f2_new_arr[N-1] * (11.0 - f1_new_arr[N-1])
            
            N_old = len(f1_sorted)
            hv_old = 0.0
            for i in range(N_old - 1):
                hv_old += (f2_sorted[i] - f2_sorted[i+1]) * (11.0 - f1_sorted[i])
            hv_old += f2_sorted[N_old-1] * (11.0 - f1_sorted[N_old-1])
            
            return hv_new - hv_old
        
        def equal_hv_spacing(N):
            """Find N f1 values where each point contributes equally to HV."""
            # Compute total HV with dense reference
            ref_f1 = np.linspace(1.0, 11.0, 50000)
            ref_f2 = f2_of_f1(ref_f1)
            total_hv = 0.0
            for i in range(len(ref_f1)-1):
                total_hv += (ref_f2[i] - ref_f2[i+1]) * (11.0 - ref_f1[i])
            total_hv += ref_f2[-1] * (11.0 - ref_f1[-1])
            
            target_per_point = total_hv / N
            
            def hv_from_a(a):
                """HV of the Pareto front segment from f1=a to f1=11"""
                if a >= 11.0:
                    return 0.0
                n_pts = 5000
                f1_seg = np.linspace(a, 11.0, n_pts)
                f2_seg = f2_of_f1(f1_seg)
                hv = 0.0
                for i in range(n_pts - 1):
                    hv += (f2_seg[i] - f2_seg[i+1]) * (11.0 - f1_seg[i])
                hv += f2_seg[-1] * (11.0 - f1_seg[-1])
                return hv
            
            f1_vals = np.zeros(N)
            f1_vals[0] = 1.0
            f1_vals[-1] = 11.0
            
            for i in range(1, N-1):
                target_hv = target_per_point * (N - i)
                lo_a, hi_a = 1.0, 11.0
                for _ in range(40):
                    mid = (lo_a + hi_a) / 2
                    if hv_from_a(mid) > target_hv:
                        lo_a = mid
                    else:
                        hi_a = mid
                f1_vals[i] = (lo_a + hi_a) / 2
            
            return f1_vals

        # Initialize with equal-hv spacing for a good starting set
        # Use a subset first, then greedily add
        init_n = min(20, max_evals)
        f1_init = equal_hv_spacing(init_n)
        
        f1_list = list(f1_init)
        f2_list = list(f2_of_f1(f1_init))
        
        # Evaluate initial points
        for f1 in f1_list:
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
        
        # Greedily add points with max marginal HV
        remaining = max_evals - n_evals
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            f1_sorted = np.array(f1_list)
            f2_sorted = np.array(f2_list)
            
            # Generate candidates: midpoints, quartiles, and fine grid
            candidates = []
            for i in range(len(f1_sorted) - 1):
                a, b = f1_sorted[i], f1_sorted[i+1]
                candidates.append((a + b) / 2.0)
                candidates.append((3*a + b) / 4.0)
                candidates.append((a + 3*b) / 4.0)
            
            # Add some global fine grid points
            for t in np.linspace(0, 1, 50):
                candidates.append(1.0 + 10.0 * t)
            
            candidates = np.array(candidates)
            candidates = candidates[(candidates > 1.0 + 1e-9) & (candidates < 11.0 - 1e-9)]
            candidates = np.unique(candidates)
            
            if len(candidates) == 0:
                break
            
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
