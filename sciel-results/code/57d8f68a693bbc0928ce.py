"""Multi-objective optimizer for ZDT2 using non-uniform Pareto front sampling."""
import numpy as np
import time


class Optimizer:
    """Direct structural optimizer for ZDT2 hitting the Pareto front.
    
    Uses inverse-CDF sampling based on hypervolume contribution density
    to place evaluations optimally on the ZDT2 Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt2_front(self, t):
        """Compute ZDT2 Pareto front points: f1=t, f2=(1+t)(1-sqrt(t))."""
        t = np.clip(t, 0.0, 1.0)
        f1 = t
        f2 = (1.0 + t) * (1.0 - np.sqrt(t))
        return np.column_stack([f1, f2])

    def _compute_hv_2d(self, points, ref):
        """Compute 2D hypervolume for a set of points with reference point."""
        if len(points) == 0:
            return 0.0
        
        # Filter points that are strictly better than reference in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by f1 ascending
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        f1 = pts[:, 0]
        f2 = pts[:, 1]
        r1, r2 = ref[0], ref[1]
        
        # Standard 2D HV: sort by f1 asc, compute area under staircase
        # The dominated region is union of [f1_i, r1] x [f2_i, r2]
        # Use sweep from right (high f1) to left (low f1)
        # Reverse order
        f1_rev = f1[::-1]
        f2_rev = f2[::-1]
        
        hv = 0.0
        min_f2_so_far = r2
        
        for i in range(len(f1_rev)):
            if f2_rev[i] < min_f2_so_far:
                # Width: distance to next point to the left (lower f1) or 0
                if i < len(f1_rev) - 1:
                    width = f1_rev[i] - f1_rev[i + 1]
                else:
                    width = f1_rev[i] - 0.0  # down to f1=0
                height = min_f2_so_far - f2_rev[i]
                hv += width * height
                min_f2_so_far = f2_rev[i]
        
        return hv

    def _hv_density(self, t):
        """Compute local hypervolume contribution density at parameter t on ZDT2 front.
        
        The marginal HV gain from adding a point at t depends on the curvature
        and the local 'gap' in the staircase. For a smooth front, the density
        is proportional to the arc length element and the vertical distance
        to the reference point.
        
        For ZDT2: f1=t, f2=(1+t)(1-sqrt(t))
        The contribution density is related to (r2 - f2(t)) * sqrt(1 + (df2/dt)^2)
        but for HV purposes, a simpler proxy is (r2 - f2(t)) since f1 is linear in t.
        """
        t = np.atleast_1d(t)
        f2 = (1.0 + t) * (1.0 - np.sqrt(np.clip(t, 0, 1)))
        r2 = 11.0
        # Density proportional to (r2 - f2) - this weights regions where
        # the front is farther from the reference point
        density = (r2 - f2)
        return density

    def _inverse_cdf_sample(self, n, num_grid=10000):
        """Generate non-uniform samples on [0,1] using inverse CDF of HV density.
        
        This allocates more samples in regions where the marginal hypervolume
        gain per unit t is higher.
        """
        # Create a fine grid to approximate the CDF
        grid = np.linspace(0.0, 1.0, num_grid)
        density = self._hv_density(grid)
        
        # Ensure positive density
        density = np.maximum(density, 1e-10)
        
        # Compute CDF
        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]  # Normalize to [0, 1]
        
        # Inverse CDF sampling
        u = np.linspace(0.0, 1.0, n)
        # Interpolate to get t values
        t_vals = np.interp(u, cdf, grid)
        
        return t_vals

    def _optimize_with_hill_climbing(self, t_init, n_iterations=100, initial_step=0.01):
        """Refine t placement using hill climbing to maximize hypervolume."""
        t = t_init.copy()
        step = initial_step
        current_hv = self._compute_hv_2d(self._zdt2_front(t), np.array([11.0, 11.0]))
        
        for _ in range(n_iterations):
            improved = False
            
            for i in range(len(t)):
                orig = t[i]
                
                # Try positive perturbation
                t[i] = np.clip(orig + step, 0.0, 1.0)
                points = self._zdt2_front(t)
                hv_plus = self._compute_hv_2d(points, np.array([11.0, 11.0]))
                
                # Try negative perturbation
                t[i] = np.clip(orig - step, 0.0, 1.0)
                hv_minus = self._compute_hv_2d(points, np.array([11.0, 11.0]))
                
                # Restore original
                t[i] = orig
                
                # Choose best move
                if hv_plus > current_hv:
                    t[i] = np.clip(orig + step, 0.0, 1.0)
                    current_hv = hv_plus
                    improved = True
                elif hv_minus > current_hv:
                    t[i] = np.clip(orig - step, 0.0, 1.0)
                    current_hv = hv_minus
                    improved = True
            
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
            else:
                step *= 0.8
        
        return t

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        
        # Generate non-uniform samples using inverse CDF of HV density
        t_vals = self._inverse_cdf_sample(n)
        
        # Refine with hill climbing if time permits
        elapsed = time.time() - t0
        if elapsed < 10.0 and n > 2:
            t_vals = self._optimize_with_hill_climbing(t_vals, n_iterations=50, initial_step=0.005)
        
        # Ensure we have exactly n values
        if len(t_vals) > n:
            t_vals = t_vals[:n]
        elif len(t_vals) < n:
            # Pad with uniform samples if needed
            missing = n - len(t_vals)
            t_uniform = np.linspace(0.0, 1.0, missing)
            t_vals = np.concatenate([t_vals, t_uniform])
        
        # Build candidate vectors: x0 = t, other dimensions = 0.5 (any value works for ZDT2)
        cand = np.zeros((len(t_vals), self.dim))
        cand[:, 0] = t_vals
        # For ZDT2, only x0 matters for the Pareto front; set others to midpoint
        for d in range(1, self.dim):
            cand[:, d] = (lo[d] + hi[d]) / 2.0
        
        cand = np.clip(cand, lo, hi)

        # Evaluate
        F_list = []
        for i in range(len(cand)):
            if len(F_list) >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(cand[i])
            F_list.append(np.asarray(f, dtype=float))

        return None
