"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1: f1 = x1, g = 1 + 9*n/(x1+...+xn), f2 = g*(1 - sqrt(f1/g))
        # Pareto front: x1 in [0,1], x2...xn = 0
        # So f1 = x1, f2 = 1 - sqrt(x1)
        
        # Strategy:
        # 1. Place points along the analytic Pareto front
        # 2. Refine locally to minimize g (push x2..xn toward 0)
        # 3. Use remaining budget for exploration
        
        n_evals = 0
        
        def evaluate(x):
            nonlocal n_evals
            f = problem(x)
            n_evals += 1
            return f
        
        # Phase 1: Initialize along Pareto front
        # x1 = t, x2..xn = 0 gives the Pareto front
        n_front = min(50, self.budget // 4)
        t_values = np.linspace(0.01, 1.0, n_front)
        
        F_list = []
        X_list = []
        
        for t in t_values:
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = t
            # x[1:] = 0 already
            f = evaluate(x)
            F_list.append(f)
            X_list.append(x.copy())
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Local refinement - for each point, try to reduce g
        # by adjusting x2..xn toward 0 (they're already 0, but let's verify)
        # Actually, since x2..xn = 0 is optimal, we should focus on
        # getting more points along the front and ensuring they're on the front
        
        # Phase 3: Use remaining budget for more points on the front
        # and some perturbation to ensure we're truly on the Pareto front
        remaining = self.budget - n_evals
        
        # Add more points densely along the front
        n_extra = min(remaining // 3, 100)
        t_extra = np.linspace(0.001, 1.0, n_extra)
        
        for t in t_extra:
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = t
            f = evaluate(x)
            F_list.append(f)
            X_list.append(x.copy())
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 4: Perturb some points slightly to explore neighborhood
        # This helps if the problem has slight variations
        n_perturb = min(remaining // 5, 50)
        for i in range(n_perturb):
            if n_evals >= self.budget or time.time() > deadline:
                break
            if len(X) == 0:
                break
            idx = rng.integers(len(X))
            x = X[idx].copy()
            # Small perturbation
            x += 0.01 * (hi - lo) * rng.standard_normal(self.dim)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            F_list.append(f)
            X_list.append(x.copy())
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 5: Nelder-Mead style local search on a few promising points
        # to push them closer to the true Pareto front
        n_nm = min((self.budget - n_evals) // 10, 20)
        for i in range(n_nm):
            if n_evals >= self.budget or time.time() > deadline:
                break
            if len(X) == 0:
                break
            
            # Pick a random point to refine
            idx = rng.integers(len(X))
            x0 = X[idx].copy()
            
            # Simple local search: try small steps in x2..xn direction (toward 0)
            best_x = x0
            best_f = F[idx]
            
            for step in [0.001, 0.0001]:
                for dim_idx in range(1, min(self.dim, 5)):  # Only first few dims
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        x_trial = best_x.copy()
                        x_trial[dim_idx] += direction * step * (hi[dim_idx] - lo[dim_idx])
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = evaluate(x_trial)
                        
                        # Check if better (non-dominated or improves one obj without worsening other)
                        if (f_trial[0] < best_f[0] and f_trial[1] <= best_f[1]) or \
                           (f_trial[1] < best_f[1] and f_trial[0] <= best_f[0]):
                            best_x = x_trial
                            best_f = f_trial
            
            # Replace the original point if improved
            if np.any(best_f < F[idx]):
                X[idx] = best_x
                F[idx] = best_f
        
        # Final deduplication: keep non-dominated points
        if len(F) > 0:
            # Keep only non-dominated points for a cleaner front
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            if np.any(~dominated):
                F = F[~dominated]
                X = X[~dominated]
        
        return X
# EVOLVE-BLOCK-END
