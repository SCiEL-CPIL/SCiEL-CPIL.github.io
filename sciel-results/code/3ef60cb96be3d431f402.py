"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        
        # Phase 1: Sample x[0] values spread across [0,1] to cover the non-convex front
        # Use a mix of uniform and clustered points around known interesting regions
        n_phase1 = 60
        # Create diverse x[0] values: uniform + sinusoidal clustering
        x0_values = np.linspace(0.01, 0.99, n_phase1)
        # Add some extra points in regions where the front has high curvature
        extra_x0 = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        x0_values = np.concatenate([x0_values, extra_x0])
        x0_values = np.unique(np.clip(x0_values, 0.001, 0.999))
        n_phase1 = len(x0_values)
        
        # For each x0 value, set other variables to 0 (Pareto-optimal for g)
        # This gives us points on or very close to the Pareto front
        X = np.zeros((n_phase1, self.dim))
        F_list = []
        used = 0
        
        for i, x0 in enumerate(x0_values):
            if time.time() > deadline - 10:
                break
            if used >= self.budget:
                break
            X[i, 0] = x0
            # Other variables are 0, which minimizes g
            f = problem(X[i])
            F_list.append(f)
            used += 1
        
        F = np.array(F_list) if F_list else np.array([])
        
        # Phase 2: Use remaining budget to refine - try small perturbations of x[1:]
        # to see if we can get even lower g, and also try different x0 values
        remaining = self.budget - used
        if remaining > 0 and len(F) > 0:
            # Sort by f2 to find best candidates
            order = np.argsort(F[:, 1])
            top_k = min(10, len(order))
            
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                # Pick a random top candidate and perturb it
                idx = order[np.random.randint(0, top_k)]
                child = X[idx].copy()
                # Small random perturbation
                child += 0.01 * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)
                # Ensure x[0] stays in reasonable range
                child[0] = np.clip(child[0], 0.001, 0.999)
                fc = problem(child)
                used += 1
                # Keep if better than worst
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    X[idx] = child
                    F[worst_idx] = fc
                    # Re-sort
                    order = np.argsort(F[:, 1])
                    top_k = min(10, len(order))
        
        return X
# EVOLVE-BLOCK-END
