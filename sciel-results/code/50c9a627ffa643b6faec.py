# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 exploiting separability and Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Track all evaluations
        all_x = []
        all_f = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_x.append(x)
            all_f.append(f)
            eval_count += 1
            return x, f

        def is_deadline():
            return time.time() > deadline

        def non_dominated_sort(F):
            # Return indices of non-dominated points
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Phase 1: Coarse sweep of x1
        # WFG4 Pareto front is determined by x1. x2..xn should be optimized to minimize g.
        # g is minimized near x2..xn = 0.
        
        n_sweep = 50
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        best_x_sweep = np.zeros((n_sweep, dim))
        best_f_sweep = np.zeros((n_sweep, self.n_obj))
        
        for i, x1_val in enumerate(x1_vals):
            if is_deadline() or eval_count >= budget:
                break
            x = np.zeros(dim)
            x[0] = x1_val
            x[1:] = 0.0
            
            x, f = evaluate(x)
            best_x_sweep[i] = x
            best_f_sweep[i] = f

        # Find non-dominated in best_f_sweep
        nd_indices = non_dominated_sort(best_f_sweep)
        nd_indices_sorted = nd_indices[np.argsort(best_f_sweep[nd_indices, 0])]
        
        # Phase 2: Local Refinement
        # Refine the non-dominated points to push x2..xn closer to the optimum for g.
        
        n_refine = min(20, len(nd_indices_sorted))
        
        for idx in nd_indices_sorted[:n_refine]:
            if is_deadline() or eval_count >= budget:
                break
            
            x_curr = best_x_sweep[idx].copy()
            f_curr = best_f_sweep[idx].copy()
            
            # Coordinate Descent on x[1:]
            step_size = 0.05 * (hi[1] - lo[1])
            for _iter in range(3):
                if is_deadline() or eval_count >= budget:
                    break
                
                improved = False
                for i in range(1, dim):
                    if is_deadline() or eval_count >= budget:
                        break
                    
                    for delta in [step_size, -step_size]:
                        x_try = x_curr.copy()
                        x_try[i] = np.clip(x_try[i] + delta, lo[i], hi[i])
                        
                        if np.array_equal(x_try, x_curr):
                            continue
                        
                        x_try, f_try = evaluate(x_try)
                        
                        # Accept if it dominates current or is incomparable but better in one obj
                        if np.all(f_try <= f_curr):
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            break
                        elif np.any(f_try < f_curr) and np.all(f_try <= f_curr + 1e-9):
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            break
                        
                step_size *= 0.5

            # Update local best (though global list already has the points)
            best_x_sweep[idx] = x_curr
            best_f_sweep[idx] = f_curr

        return all_x
# EVOLVE-BLOCK-END
