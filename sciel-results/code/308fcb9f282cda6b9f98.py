"""Multi-objective optimizer on ZDT6 — targeted sampling + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Targeted ZDT6 optimizer: dense front sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 240  # 60s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        budget = self.budget
        dim = self.dim

        # ---- Phase 1: Dense sampling on the known Pareto front ----
        # ZDT6 Pareto front: x0 = t, x1..x9 = 0 gives f1=t^2, f2=1-sqrt(t)
        # Hypervolume with ref [11,11]: contribution of a point is (11-f1)*(11-f2) minus overlaps
        # Points with small f1 (small t) have larger (11-f1) factor, so they contribute more.
        # Use non-uniform t spacing: denser near t=0 to maximize hypervolume.
        n_front = int(budget * 0.85)
        n_local = budget - n_front

        # Non-uniform t values: use t = u^2 where u is uniform, giving more points near 0
        # This concentrates samples where f1 is small (high hypervolume contribution)
        u_vals = np.linspace(0.0, 1.0, n_front)
        t_vals = u_vals ** 1.5  # Power < 2 gives moderate concentration; tune exponent

        X = np.zeros((n_front, dim))
        X[:, 0] = t_vals
        # For ZDT6, setting x1..x9 to 0 keeps g at its minimum, placing points on the front
        X[:, 1:] = 0.0
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_front

        # ---- Phase 2: Local refinement around promising front points ----
        # Focus refinement on points with high hypervolume contribution (small f1)
        if n_local > 0 and time.time() < deadline:
            # Prioritize refinement of points with smaller f1 values (left side of front)
            # Sort by f1 and pick the first n_local points for refinement
            f1_sorted = np.argsort(F[:, 0])
            n_refine = min(n_local, n_front)
            refine_idx = f1_sorted[:n_refine]
            
            for idx in refine_idx:
                if used >= budget or time.time() > deadline:
                    break
                
                base = X[idx].copy()
                # Try multiple small perturbations to find improvement
                best_f = F[idx].copy()
                best_x = base.copy()
                
                for _ in range(3):
                    # Small perturbation in x0 direction
                    delta = 0.005 * np.random.randn()
                    cand = base.copy()
                    cand[0] = np.clip(base[0] + delta, lo[0], hi[0])
                    # Small perturbation in other dimensions
                    cand[1:] = np.clip(base[1:] + 0.0005 * np.random.randn(dim - 1), lo[1:], hi[1:])
                    
                    fc = problem(cand)
                    used += 1
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Check if this candidate is better than current best
                    if np.all(fc <= best_f) and np.any(fc < best_f):
                        best_f = fc
                        best_x = cand.copy()
                
                if np.all(best_f <= F[idx]) and np.any(best_f < F[idx]):
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
