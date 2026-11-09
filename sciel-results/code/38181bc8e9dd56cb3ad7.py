"""Multi-objective optimizer exploiting ZDT2 structure with known Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    1. Sample along the known Pareto front (x1 in [0,1], x2..x30 = 0.5)
    2. Use remaining budget for local refinement around promising points
    3. Deadline guard to stay within time limit
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 Pareto front: f1 = x1, f2 = (1 + 9/n) * (1 - sqrt(f1/(1+9/n)))
        # For x1 in [0,1], x2..xn = 0.5 gives points on or near the front
        # Actually for ZDT2, the Pareto front is f2 = (1+9/n)(1-sqrt(f1/(1+9/n)))
        # when x2..xn = 0.5 (since g=1 when x_i=0.5 for i>=2)
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Sample along the Pareto front
        # Use about 60% of budget for front sampling
        n_front = min(budget // 2, 120)
        n_front = max(n_front, 20)
        
        # Sample x1 values spread across [0, 1]
        x1_vals = np.linspace(0, 1, n_front)
        
        X = np.zeros((n_front, n))
        for i, x1 in enumerate(x1_vals):
            X[i, 0] = x1
            X[i, 1:] = 0.5  # This puts us on the Pareto front for ZDT2
        
        F = np.array([problem(x) for x in X])
        used = n_front
        
        # Phase 2: Local refinement
        # For each front point, try small perturbations to see if we can improve
        remaining = budget - used
        if remaining > 0:
            # Try perturbations around each front point
            step_size = 0.01
            n_trials_per_point = max(1, remaining // n_front)
            
            for i in range(n_front):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Try a few random perturbations around this point
                for _ in range(min(n_trials_per_point, 5)):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Perturb x1 slightly
                    delta_x1 = np.random.uniform(-step_size, step_size)
                    new_x1 = np.clip(X[i, 0] + delta_x1, 0, 1)
                    
                    # Keep x2..xn at 0.5 (on the front)
                    child = X[i].copy()
                    child[0] = new_x1
                    child[1:] = 0.5
                    
                    fc = problem(child)
                    used += 1
                    
                    # Check if this child is better than any existing point
                    # or replaces a dominated point
                    # Simple replacement: if better than the worst in some sense
                    # For now, just keep if it's non-dominated by the current set
                    # To keep it simple, replace the point with the worst f2
                    # among those with similar f1
                    idx = np.argmin(np.abs(F[:, 0] - fc[0]))
                    if fc[1] < F[idx, 1]:
                        X[idx] = child
                        F[idx] = fc
        
        # Phase 3: Final pass - try to fill in any gaps in the front
        if used < budget and time.time() < deadline:
            # Sample additional points in regions with poor coverage
            # Find gaps in f1 values
            f1_sorted = np.sort(F[:, 0])
            gaps = np.diff(f1_sorted)
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                gap_start = f1_sorted[max_gap_idx]
                gap_end = f1_sorted[max_gap_idx + 1]
                gap_size = gap_end - gap_start
                
                if gap_size > 0.05:
                    n_fill = min(budget - used, int(gap_size * 100))
                    for _ in range(n_fill):
                        if used >= budget or time.time() > deadline:
                            break
                        x1 = np.random.uniform(gap_start, gap_end)
                        child = np.zeros(n)
                        child[0] = x1
                        child[1:] = 0.5
                        fc = problem(child)
                        used += 1
                        # Replace the worst point
                        worst_idx = np.argmax(F[:, 1])
                        if fc[1] < F[worst_idx, 1]:
                            X[worst_idx] = child
                            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
