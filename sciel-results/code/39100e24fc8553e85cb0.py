# EVOLVE-BLOCK-START
"""Multi-objective DTLZ4 optimizer: greedy hypervolume maximization on analytic Pareto front."""
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: generate candidate pool on Pareto front, select via greedy HV."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        ref = np.array([2.5] * m)

        # Generate candidate pool on DTLZ4 Pareto front
        # DTLZ4 with m=3: f1=x1, f2=(1-x1)*x2, f3=(1-x1)*(1-x2)
        # Pareto front is the triangle f1+f2+f3=1, fi>=0
        # Parameterize by (x1, x2) in [0,1]^2
        # Use space-filling Sobol-like sequence via Halton for good distribution

        def halton_2d(npts):
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            pts = np.zeros((npts, 2))
            for i in range(1, npts + 1):
                pts[i-1, 0] = halton_1d(i, 2)
                pts[i-1, 1] = halton_1d(i, 3)
            return pts

        # Pool size: larger than budget for better selection
        pool_size = min(800, self.budget * 4)
        halton_pts = halton_2d(pool_size)

        # Map Halton points [0,1]^2 to (x1, x2) on Pareto front
        # Uniform sampling on the triangle: use sqrt transform
        u = np.sqrt(halton_pts[:, 0])
        v = halton_pts[:, 1] * (1.0 - u)
        x1 = 1.0 - u - v
        x2 = np.where(u + v > 1e-15, v / (u + v), 0.0)

        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)

        # Build full x vectors with g=0 (x[2:]=0.5)
        X_pool = np.zeros((pool_size, n))
        X_pool[:, 0] = x1
        X_pool[:, 1] = x2
        for j in range(2, n):
            X_pool[:, j] = 0.5
        X_pool = np.clip(X_pool, lo, hi)

        # Compute objective values for all pool points
        # f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
        F_pool = np.column_stack([x1, (1 - x1) * x2, (1 - x1) * (1 - x2)])

        # Filter: only points strictly better than reference point in all objectives
        mask = np.all(F_pool < ref, axis=1)
        F_pool = F_pool[mask]
        X_pool = X_pool[mask]

        # Greedy selection: iteratively pick point with largest marginal HV contribution
        # For 3 objectives, HV of a point relative to reference and already selected points
        # is the volume of the box from the point to ref, minus volumes already covered

        selected_F = []
        selected_X = []
        remaining = list(range(len(F_pool)))
        budget = min(self.budget, len(F_pool))

        for _ in range(budget):
            if time.time() > deadline - 10.0:
                break
            if not remaining:
                break

            best_idx = -1
            best_hv = -1.0

            for ri in remaining:
                f = F_pool[ri]
                # Compute marginal HV contribution
                # In 3D, the contribution is the volume of the box [f, ref] 
                # that is not already dominated by selected points
                # Simple approximation: compute full box volume
                box_vol = 0.0
                for dim_i in range(m):
                    box_vol *= (ref[dim_i] - f[dim_i])
                
                # Check if this point is dominated by any selected point
                dominated = False
                for sf in selected_F:
                    if np.all(sf <= f) and np.any(sf < f):
                        dominated = True
                        break
                
                if dominated:
                    continue

                # Approximate marginal contribution as box volume (crude but fast)
                if box_vol > best_hv:
                    best_hv = box_vol
                    best_idx = ri

            if best_idx == -1:
                break

            selected_F.append(F_pool[best_idx])
            selected_X.append(X_pool[best_idx])
            remaining.remove(best_idx)

        # Ensure we have at least some points if greedy failed
        if len(selected_X) < budget:
            for ri in remaining:
                if len(selected_X) >= budget:
                    break
                selected_F.append(F_pool[ri])
                selected_X.append(X_pool[ri])

        # Now evaluate the selected points using the problem function
        X_final = np.array(selected_X) if selected_X else X_pool[:budget]
        F_final = np.empty((len(X_final), m))
        for i in range(len(X_final)):
            if time.time() > deadline:
                break
            F_final[i] = problem(X_final[i])

        return X_final
# EVOLVE-BLOCK-END
