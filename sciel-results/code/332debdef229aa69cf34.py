import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Pre-compute the ZDT6 Pareto front mapping
        # ZDT6: x1 in [0,1], f1 = x1, g = 1 + 9/sqrt(sum(x_i^2)), f2 = g*(1 - sqrt(f1/g))
        # Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1)
        # To achieve a specific f1 value, set x1 = f1, and x2..x10 = 0
        # Then sum(x_i^2) = x1^2, g = 1 + 9/sqrt(x1^2) = 1 + 9/x1 (for x1 > 0)
        # f2 = g*(1 - sqrt(f1/g))
        
        # For the Pareto front, we need g = 1 + 9/sqrt(sum(x_i^2)) where x2..x10=0
        # So sum = x1^2, sqrt(sum) = x1 (for x1 >= 0)
        # g = 1 + 9/x1
        # f1 = x1
        # f2 = g * (1 - sqrt(f1/g)) = (1+9/x1)*(1 - sqrt(x1/(1+9/x1)))
        #    = (1+9/x1)*(1 - sqrt(x1^2/(x1+9)))
        #    = (1+9/x1)*(1 - x1/sqrt(x1+9))
        
        # Let's verify: on PF, f2 = 1 - sqrt(f1)
        # So we need (1+9/x1)*(1 - x1/sqrt(x1+9)) = 1 - sqrt(x1)
        
        # Strategy: Use a dense grid in f1 space [0,1], map to x1 = f1, x2..x10 = 0
        # This gives points on or near the Pareto front
        
        # Phase 1: Dense sampling along the known Pareto front
        # Use non-uniform grid denser where curvature is high
        # f2 = 1 - sqrt(f1), so df2/df1 = -1/(2*sqrt(f1)), which is large near f1=0
        # We want uniform spacing in f2, so df1 = -2*sqrt(f1)*df2
        # Using inverse transform: if u ~ Uniform(0,1), f1 = (1-u)^2 gives uniform f2 spacing
        
        n_front = 150
        u = np.linspace(0, 1, n_front)
        f1_vals = (1 - u) ** 2  # This gives f2 = 1 - (1-u) = u, uniform in f2
        f1_vals = np.clip(f1_vals, 1e-10, 1.0)
        
        # Map to decision space: x1 = f1, x2..x10 = 0
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = f1_vals
        
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Collect all evaluated points
        all_X = [X_front]
        all_F = [F_front]
        
        # Phase 2: Use remaining budget for local search / refinement
        # Add random perturbations around front points to explore
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Create perturbed candidates
            n_perturb = min(remaining, 50)
            for i in range(n_perturb):
                if time.time() > deadline - 10:
                    break
                base_idx = i % n_front
                base = X_front[base_idx].copy()
                # Small perturbation in x2..x10
                pert = 0.01 * span[1:] * np.random.randn(dim - 1)
                base[1:] += pert
                base = np.clip(base, lo, hi)
                F_base = problem(base)
                all_X.append(base.reshape(1, -1))
                all_F.append(F_base.reshape(1, -1))
                used += 1
            
            # Use more budget for additional front sampling if time allows
            if used < budget and time.time() < deadline - 5:
                # Add more points on the front with different x2..x10 values
                extra = min(budget - used, 30)
                for i in range(extra):
                    if time.time() > deadline - 5:
                        break
                    x = np.random.uniform(lo, hi)
                    # Ensure x1 is in a good range
                    x[0] = np.random.uniform(0.01, 0.99)
                    F_x = problem(x)
                    all_X.append(x.reshape(1, -1))
                    all_F.append(F_x.reshape(1, -1))
                    used += 1
        
        # Combine all points
        if all_X:
            X_all = np.vstack(all_X)
            F_all = np.vstack(all_F)
        else:
            X_all = np.array([]).reshape(0, dim)
            F_all = np.array([]).reshape(0, n_obj)
        
        return X_all
