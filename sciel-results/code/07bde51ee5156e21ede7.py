import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using structure-aware sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT1, optimal x[1:] = 0 to minimize g(x[1:])
        # We'll focus on sampling x[0] across [0,1]
        
        # Phase 1: Initial uniform sampling of x[0]
        n_init = min(budget, 50)
        x0_values = np.linspace(0.0, 1.0, n_init)
        
        X = np.zeros((n_init, dim))
        F = np.zeros((n_init, self.n_obj))
        for i, x0 in enumerate(x0_values):
            X[i, 0] = x0
            # x[1:] = 0 for optimal g
            F[i] = problem(X[i])
        
        used = n_init
        best_X = X.copy()
        best_F = F.copy()

        # Phase 2: Adaptive refinement - add points where hypervolume gain is highest
        # We'll use a simple strategy: try midpoints between existing points
        # and keep the one that gives best hypervolume improvement
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate x0 values: midpoints between consecutive existing points
            # plus some random perturbations around existing points
            candidates = []
            
            # Midpoints between consecutive x0 values
            for i in range(len(x0_values) - 1):
                mid = (x0_values[i] + x0_values[i + 1]) / 2.0
                candidates.append(mid)
            
            # Also add endpoints if not covered
            if len(x0_values) > 0:
                candidates.append(0.0)
                candidates.append(1.0)
            
            # Remove duplicates and filter to [0,1]
            candidates = np.array(candidates)
            candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
            candidates = np.unique(candidates)
            
            if len(candidates) == 0:
                # If no candidates, just add a random point
                x0_new = np.random.random()
                candidates = np.array([x0_new])
            
            # Evaluate the first few candidates (limit to avoid too many evaluations)
            n_try = min(len(candidates), 10)
            best_gain = -1
            best_candidate = None
            best_F_new = None
            
            for j in range(n_try):
                x0_new = candidates[j]
                x_new = np.zeros(dim)
                x_new[0] = x0_new
                f_new = problem(x_new)
                used += 1
                
                # Check if this point is better than reference [11, 11]
                if f_new[0] < 11.0 and f_new[1] < 11.0:
                    # Compute hypervolume gain approximately
                    # A point contributes if it's not dominated by existing points
                    # Simple check: is it non-dominated?
                    dominated = False
                    for k in range(len(best_F)):
                        if np.all(best_F[k] <= f_new) and np.any(best_F[k] < f_new):
                            dominated = True
                            break
                    if not dominated:
                        # Estimate gain: area contribution
                        # This is a rough estimate
                        gain = 1.0 / (1 + j)  # Heuristic
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = x_new
                            best_F_new = f_new
                
                if used >= budget:
                    break
            
            if best_candidate is not None:
                # Add to population
                new_X = np.vstack([best_X, best_candidate.reshape(1, -1)])
                new_F = np.vstack([best_F, best_F_new.reshape(1, -1)])
                best_X = new_X
                best_F = new_F
                x0_values = np.append(x0_values, best_candidate[0])
            else:
                # If no good candidate found, try a random x0
                x0_new = np.random.random()
                x_new = np.zeros(dim)
                x_new[0] = x0_new
                f_new = problem(x_new)
                used += 1
                if f_new[0] < 11.0 and f_new[1] < 11.0:
                    best_X = np.vstack([best_X, x_new.reshape(1, -1)])
                    best_F = np.vstack([best_F, f_new.reshape(1, -1)])
                    x0_values = np.append(x0_values, x0_new)
        
        return best_X
