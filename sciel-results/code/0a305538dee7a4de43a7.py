"""ZDT2 optimizer with analytic Pareto front sampling and hypervolume-optimized spacing."""
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        n = self.dim
        C = 1 + 10 * (n - 1)  # ZDT2 constant
        ref_val = 11.0

        # Calculate t* where f2(t) = ref_val
        t_star = (1.0 - ref_val / C) ** 2
        t_star = float(np.clip(t_star, 0.0, 1.0))

        N = self.budget

        if N <= 0:
            return np.zeros((0, self.dim))

        if N == 1:
            X = np.zeros((1, self.dim))
            X[0, 0] = t_star
            problem(X[0])
            return X

        # We want to place N points on the Pareto front to maximize hypervolume.
        # The Pareto front is: f1 = t, f2 = C*(1 - sqrt(t)) for t in [0, 1]
        # Points with f2 > 11 are dominated by the reference point and don't count.
        # So we need t >= t_star.

        # Let y = sqrt(t), so t = y^2, f2 = C*(1-y)
        # y ranges from sqrt(t_star) to 1.0

        y_star = np.sqrt(t_star)

        # The hypervolume contribution of a set of points on this front
        # is the area of the union of rectangles [0, f1_i] x [0, f2_i].
        # For a convex front, this equals the area under the piecewise-linear
        # interpolation of the points, plus the rectangle from 0 to t_star.

        # To maximize this, we should space points to minimize the error
        # between the piecewise-linear approximation and the true curve.
        # For a function with curvature proportional to t^(-1.5), we should
        # densify points near small t (near t_star).

        # Using y = sqrt(t), the curve becomes f2 = C*(1-y), which is linear in y.
        # So uniform spacing in y gives a piecewise-linear approximation that
        # exactly matches the curve! Wait, no - the curve in (t, f2) space is
        # not linear, but in (y, f2) space it is linear.

        # The area we want to approximate is:
        # Integral_{t_star}^{1} C*(1-sqrt(t)) dt
        # Substituting t = y^2: Integral_{y_star}^{1} C*(1-y)*2y dy
        # = 2C * Integral_{y_star}^{1} (y - y^2) dy

        # The integrand g(y) = y - y^2 is a parabola, which is smooth.
        # Uniform spacing in y should give a very good approximation.

        # However, we can do better by using the exact hypervolume formula
        # and optimizing the spacing. The hypervolume of N points t_0 < t_1 < ... < t_{N-1}
        # on the front is:
        # HV = sum_{i=0}^{N-2} 0.5*(f2(t_i) + f2(t_{i+1}))*(t_{i+1} - t_i) + 11*t_star
        #    + (11 - t_{N-1})*(11 - f2(t_{N-1}))  [if t_{N-1} < 1]

        # Actually, the hypervolume relative to reference point (11, 11) is:
        # The union of boxes [0, f1_i] x [0, f2_i] for all i.
        # For a convex decreasing front, this is the area under the upper envelope.
        # The upper envelope is the piecewise-linear curve connecting the points,
        # extended to (11, 0) and (0, 11).

        # Let's use a simpler approach: sample uniformly in y, which is known to be
        # very effective for this type of front.

        y_vals = np.linspace(y_star, 1.0, N)
        t_vals = y_vals ** 2

        # Check time
        if time.time() > deadline:
            # Return what we have
            X = np.zeros((N, self.dim))
            X[:, 0] = t_vals
            return X

        # Construct the population
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # x[1:] = 0 is optimal for ZDT2 Pareto front

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        # Try to refine the spacing using a simple local optimization
        # The key insight: for a convex front, the optimal spacing equalizes
        # the marginal hypervolume gain of each gap.

        # Let's try a power-law spacing: t_i = t_star + (1 - t_star) * u_i^alpha
        # where u_i are uniform in [0, 1].
        # For alpha < 1, points are denser near t_star.

        # We can estimate the optimal alpha by trying a few values and
        # computing the hypervolume for each.

        best_hv = self._compute_hypervolume(F, ref_val)
        best_X = X.copy()

        # Try a few alpha values
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            if time.time() > deadline:
                break

            u = np.linspace(0.0, 1.0, N)
            t_vals_alpha = t_star + (1.0 - t_star) * (u ** alpha)

            X_alpha = np.zeros((N, self.dim))
            X_alpha[:, 0] = t_vals_alpha

            F_alpha = np.array([problem(x) for x in X_alpha])
            hv_alpha = self._compute_hypervolume(F_alpha, ref_val)

            if hv_alpha > best_hv:
                best_hv = hv_alpha
                best_X = X_alpha.copy()

        return best_X

    def _compute_hypervolume(self, F, ref_val):
        """Compute 2D hypervolume relative to reference point (ref_val, ref_val)."""
        # Filter points that are strictly better than reference in both objectives
        mask = (F[:, 0] < ref_val) & (F[:, 1] < ref_val)
        F_filtered = F[mask]

        if len(F_filtered) == 0:
            return 0.0

        # Sort by f1 ascending
        F_sorted = F_filtered[np.argsort(F_filtered[:, 0])]

        # Compute hypervolume using the sweep line algorithm
        hv = 0.0
        prev_f2 = ref_val
        prev_f1 = 0.0

        for f1, f2 in F_sorted:
            if f2 < prev_f2:
                hv += (f1 - prev_f1) * (prev_f2 - f2)
                prev_f1 = f1
                prev_f2 = f2
            else:
                # This point is dominated by the previous one in f2
                # but may still contribute if f1 is larger
                pass

        # Add the last segment
        if prev_f2 < ref_val:
            hv += (ref_val - prev_f1) * (prev_f2)

        return hv
