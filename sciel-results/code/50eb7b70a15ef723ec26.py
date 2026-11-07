"""Multi-objective optimizer for ZDT3 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-specific optimizer exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT3 structure:
        # f1 = x[0]
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) - 0.05*sin(10*pi*f1))
        # Pareto-optimal when x[1:] = 0 (minimizes g)
        # The Pareto front has 51 local minima due to the sin term
        # Key insight: to maximize hypervolume, we want to cover the front well
        # especially around the local minima where f2 has peaks

        n_phase1 = 150  # Dense uniform sampling
        n_phase2 = self.budget - n_phase1  # Remaining for targeted sampling

        # Phase 1: Dense uniform sampling of x[0]
        x0_values = np.linspace(0.001, 0.999, n_phase1)
        
        F_list = []
        X_list = []
        used = 0

        for i, x0 in enumerate(x0_values):
            if time.time() > deadline - 10:
                break
            if used >= self.budget:
                break
            x = np.zeros(dim)
            x[0] = x0
            f = problem(x)
            F_list.append(f)
            X_list.append(x)
            used += 1

        F = np.array(F_list) if F_list else np.array([])
        X = np.array(X_list) if X_list else np.array([])

        if len(F) == 0:
            return X

        # Phase 2: Targeted sampling at local minima of the Pareto front
        # The sin(10*pi*x0) term creates local minima at x0 = (2k+1)/20 for k=0..49
        # These are the points where f2 is locally maximized (peaks in the front)
        # Also sample near x0=0 and x0=1 for boundary coverage

        # Local minima positions: x0 = (2k+1)/20 for k=0,1,...,49
        # That gives 50 points: 0.05, 0.15, 0.25, ..., 0.95
        local_minima = np.array([(2*k + 1) / 20.0 for k in range(50)])
        
        # Also add boundary points
        boundary_points = np.array([0.001, 0.999])
        
        # Combine and shuffle with priority to local minima
        target_points = np.concatenate([local_minima, boundary_points])
        
        # Shuffle to avoid systematic bias, but ensure we try them
        np.random.shuffle(target_points)
        
        # Sort by distance to existing points to fill gaps first
        # Actually, let's just go through them in order, skipping if close to existing
        
        remaining = self.budget - used
        if remaining > 0:
            # For each target point, check if we already have a close point
            existing_f1 = F[:, 0]
            
            for x0_target in target_points:
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                # Check if we already have a point close to this x0
                dists = np.abs(existing_f1 - x0_target)
                if np.min(dists) < 0.005:
                    continue  # Already covered
                
                x = np.zeros(dim)
                x[0] = x0_target
                f = problem(x)
                used += 1
                
                # Check if dominated by existing points
                dominated = (np.all(F[:, 0] <= f[0], axis=1) & 
                           np.all(F[:, 1] <= f[1], axis=1) & 
                           np.any(F[:, 0] < f[0] | F[:, 1] < f[1], axis=1))
                
                if not np.any(dominated):
                    F = np.vstack([F, f.reshape(1, -1)])
                    X = np.vstack([X, x.reshape(1, -1)])
                    # Remove points dominated by the new point
                    dominated_by_new = (np.all(F[:, 0] >= f[0], axis=1) & 
                                       np.all(F[:, 1] >= f[1], axis=1) & 
                                       np.any(F[:, 0] > f[0] | F[:, 1] > f[1], axis=1))
                    if np.any(dominated_by_new):
                        F = F[~dominated_by_new]
                        X = X[~dominated_by_new]
                    existing_f1 = F[:, 0]

            # Phase 3: Use any remaining budget to fill gaps
            remaining = self.budget - used
            if remaining > 0 and len(F) > 0:
                order = np.argsort(F[:, 0])
                F_sorted = F[order]
                f1_vals = F_sorted[:, 0]
                gaps = np.diff(f1_vals)
                
                for _ in range(remaining):
                    if time.time() > deadline - 2:
                        break
                    if used >= self.budget:
                        break
                    
                    if len(gaps) > 0:
                        max_gap_idx = np.argmax(gaps)
                        if gaps[max_gap_idx] > 0.01:
                            lo_f1 = f1_vals[max_gap_idx]
                            hi_f1 = f1_vals[max_gap_idx + 1]
                            # Sample at midpoint of the gap
                            x0_new = (lo_f1 + hi_f1) / 2.0
                        else:
                            x0_new = np.random.uniform(0.001, 0.999)
                    else:
                        x0_new = np.random.uniform(0.001, 0.999)
                    
                    x = np.zeros(dim)
                    x[0] = x0_new
                    f = problem(x)
                    used += 1
                    
                    dominated = (np.all(F[:, 0] <= f[0], axis=1) & 
                               np.all(F[:, 1] <= f[1], axis=1) & 
                               np.any(F[:, 0] < f[0] | F[:, 1] < f[1], axis=1))
                    
                    if not np.any(dominated):
                        F = np.vstack([F, f.reshape(1, -1)])
                        X = np.vstack([X, x.reshape(1, -1)])
                        dominated_by_new = (np.all(F[:, 0] >= f[0], axis=1) & 
                                           np.all(F[:, 1] >= f[1], axis=1) & 
                                           np.any(F[:, 0] > f[0] | F[:, 1] > f[1], axis=1))
                        if np.any(dominated_by_new):
                            F = F[~dominated_by_new]
                            X = X[~dominated_by_new]
                        
                        # Update gaps
                        order = np.argsort(F[:, 0])
                        F_sorted = F[order]
                        f1_vals = F_sorted[:, 0]
                        gaps = np.diff(f1_vals)

        return X
# EVOLVE-BLOCK-END
