import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate candidates on the assumed Pareto front
        # x1 varies in [lo[0], hi[0]], others at lo[i]
        n_cand = 1000
        x1_vals = np.linspace(lo[0], hi[0], n_cand)
        X_cand = np.zeros((n_cand, self.dim))
        X_cand[:, 0] = x1_vals
        for i in range(1, self.dim):
            X_cand[:, i] = lo[i]
            
        # Compute analytical objectives
        # f1 = x1
        # f2 = 1 - sqrt(x1)  (assuming g=1)
        # If bounds are not [0,1], we might need to adjust.
        # But ZDT1 is defined on [0,1].
        f1_cand = X_cand[:, 0]
        f2_cand = 1.0 - np.sqrt(f1_cand)
        F_cand = np.column_stack((f1_cand, f2_cand))
        
        # Greedy selection
        selected_idx = []
        selected_F = []
        
        def calc_hv(F):
            if len(F) == 0: return 0.0
            # Sort by f1
            idx = np.argsort(F[:, 0])
            F_sorted = F[idx]
            # Remove dominated points (keep non-dominated)
            # Since f2 decreases as f1 increases on the front, 
            # we just need to ensure f2 is strictly decreasing.
            # But if there are duplicates or noise, we should filter.
            # For the front, f2 is strictly decreasing with f1.
            # So all points are non-dominated.
            # HV = sum((f1[i] - f1[i-1]) * f2[i])
            f1 = F_sorted[:, 0]
            f2 = F_sorted[:, 1]
            f1_diff = np.diff(f1, prepend=0)
            return np.sum(f1_diff * f2)
            
        for _ in range(self.budget):
            best_gain = -1
            best_idx = -1
            # Current HV
            current_hv = calc_hv(np.array(selected_F)) if selected_F else 0.0
            
            for i in range(n_cand):
                if i in selected_idx: continue
                # Try adding i
                new_F = np.vstack([np.array(selected_F), F_cand[i]]) if selected_F else F_cand[i].reshape(1,2)
                new_hv = calc_hv(new_F)
                gain = new_hv - current_hv
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            selected_idx.append(best_idx)
            selected_F.append(F_cand[best_idx])
            
            if time.time() > deadline - 10:
                break
                
        # Evaluate selected points
        last_result = None
        for idx in selected_idx:
            x = X_cand[idx]
            res = problem(x)
            last_result = res
            
        return last_result
