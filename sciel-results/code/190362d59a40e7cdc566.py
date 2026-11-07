"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast multi-objective optimizer using local search on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Store all evaluated points to return them (or just the front? 
        # The task says "Every returned objective vector is recorded". 
        # The evaluator records the return of __call__? No, "The return value of __call__ is ignored."
        # Wait: "Every returned objective vector is recorded". This implies the evaluator calls problem(x) 
        # and records the result? No, "problem(x) ... returns 3 objective values."
        # "Every returned objective vector is recorded" likely refers to the outputs of problem(x) 
        # during the execution of __call__.
        # So we just need to make sure we call problem(x) and don't crash.
        # The score is based on the set of all points evaluated.
        
        # We will maintain a list of all evaluated points (x, f) to ensure we cover the space.
        # But we only need to return X? The return is ignored.
        # So we just need to perform the evaluations.
        
        all_X = []
        all_F = []
        
        # Initial random sampling
        n_init = min(50, budget)
        X_init = lo + span * rng.random((n_init, dim))
        for x in X_init:
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
        
        used = n_init
        
        # Maintain Pareto Front explicitly for efficiency
        # pareto_front_indices are indices into all_X/all_F
        pareto_front = list(range(n_init))
        
        def is_dominated(f_new, f_old):
            # True if f_old dominates f_new
            return np.all(f_old <= f_new) and np.any(f_old < f_new)

        def update_pareto_front(new_idx):
            f_new = all_F[new_idx]
            # Check if new point is dominated by any in front
            for i in pareto_front[:]:
                if is_dominated(f_new, all_F[i]):
                    return # New point is dominated, discard from front
            
            # If not dominated, it enters the front.
            # Remove any points from front that are dominated by the new point
            new_front = []
            for i in pareto_front:
                if not is_dominated(all_F[i], f_new):
                    new_front.append(i)
            pareto_front.clear()
            pareto_front.extend(new_front)
            pareto_front.append(new_idx)

        # Update initial front
        pareto_front.clear()
        for i in range(n_init):
            update_pareto_front(i)

        step_size = 0.1 * span
        
        while used < budget:
            if time.time() > deadline - 5.0: # Safety margin
                break
            
            # Pick a parent from the Pareto front
            if pareto_front:
                parent_idx = pareto_front[rng.integers(len(pareto_front))]
            else:
                parent_idx = rng.integers(len(all_X))
            
            parent_x = all_X[parent_idx]
            
            # Generate child with local perturbation
            # Adaptive step size: start larger, shrink over time
            progress = used / budget
            current_step = step_size * (1.0 - 0.5 * progress)
            
            child_x = parent_x + current_step * rng.standard_normal(dim)
            child_x = np.clip(child_x, lo, hi)
            
            f_child = problem(child_x)
            used += 1
            
            new_idx = len(all_X)
            all_X.append(child_x)
            all_F.append(f_child)
            
            update_pareto_front(new_idx)

        return np.array(all_X)
# EVOLVE-BLOCK-END
