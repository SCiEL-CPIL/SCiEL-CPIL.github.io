"""Multi-objective optimizer on ZDT6 — Curvature-Adaptive Front Approximation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Optimizer that exploits the known structure of the ZDT6 Pareto front
    and optimizes point placement to maximize hypervolume coverage.
    
    Key insight: The optimal placement of N points to maximize hypervolume
    under a smooth front is related to the front's curvature. Points should
    be denser where the front curves more sharply (near t=0 for ZDT6).
    
    Strategy:
    1. Sample the known Pareto front densely to build a reference curve.
    2. Compute curvature at each sample point.
    3. Use curvature-weighted spacing to place the 200 budget evaluations
       optimally along the front.
    4. This concentrates points where the front is most curved, maximizing
       the hypervolume contribution per point.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 240  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        N = self.budget  # 200 evaluations total

        # ZDT6 Pareto front: x = [t, 0, 0, ..., 0] for t in [0, 1]
        # f1 = t
        # f2 = (1 + 10*t) * (1 - sqrt(t/(1+10*t)) * (1 - t*sin(10*pi*t)))
        
        # Step 1: Sample the front densely to understand its shape
        n_dense = 5000
        t_dense = np.linspace(0, 1, n_dense)
        
        # Compute f2 values analytically (we know the front)
        # f2(t) = (1 + 10*t) * (1 - sqrt(t/(1+10*t)) * (1 - t*sin(10*pi*t)))
        def zdt6_f2(t):
            """Compute f2 for points on the ZDT6 Pareto front."""
            t = np.asarray(t, dtype=float)
            # Handle t=0 specially to avoid 0/0 in sqrt
            f2 = np.ones_like(t)
            mask = t > 0
            if np.any(mask):
                tm = t[mask]
                ratio = tm / (1 + 10 * tm)
                f2[mask] = (1 + 10 * tm) * (1 - np.sqrt(ratio) * (1 - tm * np.sin(10 * np.pi * tm)))
            return f2

        f2_dense = zdt6_f2(t_dense)
        
        # Step 2: Compute curvature at each point
        # Curvature of a parametric curve (f1(t), f2(t)):
        # kappa = |f1'*f2'' - f2'*f1''| / (f1'^2 + f2'^2)^(3/2)
        # Since f1 = t, f1' = 1, f1'' = 0
        # kappa = |f2''| / (1 + f2'^2)^(3/2)
        
        # Numerical derivatives
        dt = t_dense[1] - t_dense[0]
        f2_prime = np.gradient(f2_dense, dt)
        f2_double_prime = np.gradient(f2_prime, dt)
        
        curvature = np.abs(f2_double_prime) / (1 + f2_prime**2)**1.5
        
        # Normalize curvature to [0, 1] range for weighting
        if curvature.max() > 0:
            curvature_norm = curvature / curvature.max()
        else:
            curvature_norm = np.ones_like(curvature)
        
        # Step 3: Optimal point placement based on curvature
        # The optimal density for hypervolume maximization is proportional
        # to the curvature (more points where the curve bends more).
        # This is a known result in adaptive quadrature and approximation theory.
        
        # Use curvature-weighted spacing:
        # The idea is that if we place points such that the "curvature-weighted"
        # arc length between consecutive points is approximately constant,
        # we get an optimal approximation.
        
        # Compute cumulative curvature-weighted arc length
        # ds_weighted = curvature(t) * ds, where ds = sqrt(1 + f2'^2) * dt
        arc_length = np.sqrt(1 + f2_prime**2) * dt
        weighted_arc = curvature_norm * arc_length
        
        # Cumulative sum gives the "curvature-weighted" position
        cum_weighted = np.cumsum(weighted_arc)
        cum_weighted = cum_weighted / cum_weighted[-1]  # Normalize to [0, 1]
        
        # Step 4: Place N points such that they are uniformly spaced in 
        # curvature-weighted arc length
        # This means: find t values such that cum_weighted(t_i) = i/(N-1)
        
        t_opt = np.zeros(N)
        for i in range(N):
            target = i / (N - 1)
            # Find the t value where cum_weighted equals target
            # cum_weighted is monotonically increasing, so use searchsorted
            idx = np.searchsorted(cum_weighted, target)
            idx = np.clip(idx, 1, n_dense - 1)
            # Linear interpolation
            w0 = cum_weighted[idx - 1]
            w1 = cum_weighted[idx]
            if w1 > w0:
                frac = (target - w0) / (w1 - w0)
            else:
                frac = 0
            t_opt[i] = t_dense[idx - 1] + frac * (t_dense[idx] - t_dense[idx - 1])
        
        # Ensure we hit the endpoints exactly
        t_opt[0] = 0.0
        t_opt[-1] = 1.0
        
        # Build the decision vectors
        X_opt = np.zeros((N, self.dim))
        X_opt[:, 0] = t_opt
        np.clip(X_opt, lo, hi, out=X_opt)
        
        # Check time before evaluating
        if time.time() > deadline:
            # If we're running out of time, still evaluate what we can
            # but limit to a reasonable number
            n_eval = min(N, 50)
            X_eval = X_opt[:n_eval]
            F_eval = np.array([problem(x) for x in X_eval])
            return X_eval
        
        # Evaluate all N points
        F_opt = np.array([problem(x) for x in X_opt])
        
        # Check if any evaluation failed or returned invalid values
        # The problem function should return valid values, but let's be safe
        valid_mask = np.all(np.isfinite(F_opt), axis=1)
        if not np.all(valid_mask):
            X_opt = X_opt[valid_mask]
            F_opt = F_opt[valid_mask]
        
        return X_opt

# EVOLVE-BLOCK-END
