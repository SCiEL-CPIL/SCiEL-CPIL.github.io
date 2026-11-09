import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Latin Hypercube Sampling for initial points
        n_init = 10
        n_per_dim = n_init
        X_init = np.zeros((n_init, self.dim))
        for j in range(self.dim):
            perm = np.random.permutation(n_init)
            X_init[:, j] = (perm + 0.5) / n_init
        X_init = lo + (hi - lo) * X_init
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in X_init])
        self.eval_count += n_init
        
        # Non-dominated set
        non_dominated = list(zip(X_init, F_init))
        
        # Local search around non-dominated points
        remaining_budget = self.budget - self.eval_count
        if remaining_budget > 0:
            # We will perform local search around each non-dominated point
            # Use a simple step-based local search
            step_size = 0.1
            for _ in range(min(remaining_budget, 50)):  # Limit local search iterations
                if self.eval_count >= self.budget:
                    break
                
                # Pick a random non-dominated point
                idx = np.random.randint(len(non_dominated))
                x0, f0 = non_dominated[idx]
                
                # Generate a random direction
                direction = np.random.randn(self.dim)
                direction /= np.linalg.norm(direction)
                
                # Try a small step
                x_new = x0 + step_size * (hi - lo) * direction
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                self.eval_count += 1
                
                # Check if f_new is non-dominated
                dominated = False
                for _, f_old in non_dominated:
                    if np.all(f_old <= f_new) and np.any(f_old < f_new):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove any solutions dominated by f_new
                    non_dominated = [
                        (x, f) for x, f in non_dominated 
                        if not (np.all(f_new <= f) and np.any(f_new < f))
                    ]
                    non_dominated.append((x_new, f_new))
        
        # Return the final non-dominated set (as X, though return value is ignored)
        X_final = np.array([x for x, _ in non_dominated])
        return X_final
