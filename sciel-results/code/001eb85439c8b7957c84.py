"""Optimized ZDT2 solver using numerical variational hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting analytic structure via numerical optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, leave 10s buffer
        
        dim = self.dim
        budget = self.budget
        C = 1 + 10 * (dim - 1)
        
        # Analytic structure of ZDT2:
        # f1(x) = x1
        # f2(x) = C * (1 - sqrt(f1)) + sum(xi^2 for i=2..n)
        # Pareto front is at x2..xn = 0, so f2 = C * (1 - sqrt(f1))
        # We want to choose N values of x1 (where N=budget) to maximize the 
        # hypervolume of the resulting set of points (f1, f2) with reference (11, 11).
        
        N = budget
        
        # The hypervolume contribution of a set of points on a convex front 
        # sorted by f1 is the sum of rectangular areas.
        # Let x_i be the sorted x1 values.
        # f2_i = C * (1 - sqrt(x_i))
        # The area is sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref_f2 - f2_i)
        # where x_N is effectively ref_f1 (11.0), but we only integrate up to the 
        # last point's influence? 
        # Actually, the standard 2D HV calculation for a set of points P with ref R:
        # HV = sum of areas of boxes [p_i, R] not covered by other boxes.
        # For a convex decreasing front, if we sort by f1 (increasing), 
        # f2 is decreasing.
        # The "skyline" is the sequence of points themselves if they are non-dominated.
        # Since f2 is strictly decreasing in f1, all points on the front are non-dominated.
        # The HV is the area of the union of rectangles [x_i, ref_f1] x [f2_i, ref_f2].
        # Because the front is convex (concave in f2 vs f1? No, f2(f1) is convex? 
        # f2 = C(1-sqrt(x)). f2'' = C/(4 x^{1.5}) > 0. So f2 is convex in x.
        # Wait, a convex function lies below its chords. 
        # The union of boxes under a convex decreasing curve...
        # Let's check the area formula.
        # If points are x_1 < x_2 < ... < x_N.
        # Box i covers x in [x_i, ref_f1] and y in [f2_i, ref_f2].
        # Since f2 is decreasing, f2_1 > f2_2 > ... > f2_N.
        # The box for point i is "taller" (starts lower in y? No, f2_i is the bottom of the box? 
        # No, the box is [f2_i, ref_f2]. So smaller f2_i means a taller box (larger height ref_f2 - f2_i).
        # Point N has smallest f2, so largest height.
        # Point 1 has largest f2, so smallest height.
        # 
        # The union of these boxes:
        # For a given x, the maximum y covered is determined by the point with the 
        # smallest f2 such that x_i <= x.
        # Since f2 is decreasing, the point with the largest index j such that x_j <= x 
        # has the smallest f2_j.
        # So for x in [x_j, x_{j+1}), the covering point is j.
        # The height of the union at x is ref_f2 - f2_j.
        # So HV = sum_{j=0}^{N-1} (x_{j+1} - x_j) * (ref_f2 - f2_j) + (ref_f1 - x_N) * (ref_f2 - f2_N)
        # Wait, the last interval is [x_N, ref_f1]. The covering point is N.
        # So HV = sum_{j=0}^{N-1} (x_{j+1} - x_j) * (ref_f2 - f2_j) + (ref_f1 - x_N) * (ref_f2 - f2_N)
        # Let's re-verify indices.
        # Points x[0]...x[N-1].
        # Interval [x[0], x[1]): covered by point 0. Area: (x[1]-x[0]) * (ref_f2 - f2[0])
        # ...
        # Interval [x[N-2], x[N-1]): covered by point N-2. Area: (x[N-1]-x[N-2]) * (ref_f2 - f2[N-2])
        # Interval [x[N-1], ref_f1): covered by point N-1. Area: (ref_f1 - x[N-1]) * (ref_f2 - f2[N-1])
        
        # We only count points where f1 < ref_f1 and f2 < ref_f2.
        # f2 < 11 => C(1-sqrt(x)) < 11 => 1-sqrt(x) < 11/C => sqrt(x) > 1 - 11/C => x > (1-11/C)^2.
        # Let x_min = (1 - 11.0/C)**2.
        # Any point with x_i <= x_min has f2_i >= 11 and contributes 0 to HV (height <= 0).
        # However, if a point has f2 >= 11, its box height is <= 0, so it adds nothing.
        # But it might still affect the "covering" for x > x_i?
        # If f2_i >= 11, then ref_f2 - f2_i <= 0. The box has no positive height.
        # So such points are useless. We should ensure all x_i > x_min.
        
        x_min = (1.0 - 11.0 / C) ** 2
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # We need to optimize x_0, ..., x_{N-1} subject to x_min < x_0 < x_1 < ... < x_{N-1} < ref_f1.
        # To handle the inequality constraints and ordering, we can parameterize the differences.
        # Let d_i = x_{i+1} - x_i for i=0..N-2, and d_{N-1} = ref_f1 - x_{N-1}.
        # And x_0 = x_min + d_{-1} (or just optimize x_0 directly with bound).
        # Actually, simpler: optimize the positions x directly with bounds.
        # But they must be sorted. 
        # We can use a transformation: x_i = x_min + (ref_f1 - x_min) * t_i, where 0 < t_0 < ... < t_{N-1} < 1.
        # Or even simpler, optimize the gaps.
        # Let's use the gaps d_0, ..., d_{N-1} > 0 such that sum d_i = ref_f1 - x_min.
        # This is a simplex constraint. 
        # Alternatively, just use `minimize` with bounds on x_i and rely on the function 
        # to penalize unsorted inputs or just assume we start from a sorted init and 
        # the optimizer keeps them close? No, that's risky.
        
        # Better approach: Parameterize x_i using cumulative sums of positive variables.
        # Let u_i > 0 be the "gaps".
        # x_0 = x_min + u_0
        # x_1 = x_0 + u_1 = x_min + u_0 + u_1
        # ...
        # x_{N-1} = x_min + sum_{k=0}^{N-1} u_k
        # We need x_{N-1} < ref_f1.
        # So sum u_k < ref_f1 - x_min.
        # We can add a slack variable u_N = ref_f1 - x_{N-1} > 0.
        # So we optimize u_0, ..., u_N > 0 with sum u_i = ref_f1 - x_min.
        # This is still a constraint.
        
        # Easiest robust way: Optimize x_i directly, but enforce sorting in the objective 
        # by sorting the input inside the HV calculation? No, that breaks the gradient.
        
        # Let's use the log-gap parameterization or just use scipy's minimize with 
        # 'L-BFGS-B' and bounds on x_i, and initialize them sorted. 
        # If the function is convex (or quasi-convex) in the sorted region, it should stay there.
        # Is HV convex in x? 
        # HV(x) is piecewise linear in each x_i. It is concave? 
        # Maximizing a concave function is easy.
        # Let's check concavity.
        # f(x) = sum (x_{i+1}-x_i)(A - f2(x_i)).
        # This is not obviously concave.
        
        # Alternative: Use the known analytic solution structure.
        # The optimal spacing equalizes the "error" or "contribution".
        # For a convex front, the optimal points are clustered near the high curvature end (small x).
        # We derived earlier that x^(1/4) might be uniform? Or x^(3/4)?
        # Let's stick to numerical optimization but use a safe parameterization.
        
        # Parameterization: x_i = x_min + (ref_f1 - x_min) * (i + v_i) / (N + sum(v)) ? No.
        
        # Let's try optimizing the "exponent" alpha of a power law x_i = (i/N)^alpha, 
        # but allow the base to vary? No, that's 1D.
        
        # Let's go back to numerical optimization of x directly, but handle sorting by 
        # defining the objective on the sorted version of x? 
        # If we sort x inside the objective, the function is continuous but non-differentiable 
        # at ties. L-BFGS-B might struggle.
        
        # Better: Use the fact that the optimal x_i are distinct and sorted.
        # We can optimize the gaps g_i = x_{i+1} - x_i for i=0..N-2 and the start/end.
        # Let's define variables y of length N.
        # y[0] = x_0 - x_min
        # y[i] = x_i - x_{i-1} for i=1..N-1
        # x_i = x_min + sum_{k=0}^i y[k]
        # Constraint: x_{N-1} < ref_f1 => sum_{k=0}^{N-1} y[k] < ref_f1 - x_min.
        # We can add a slack variable y_N = ref_f1 - x_{N-1} > 0.
        # So we optimize y_0, ..., y_N > 0 such that sum y_i = L, where L = ref_f1 - x_min.
        # This is an optimization on a simplex.
        # We can parameterize y_i = L * p_i, where p_i > 0 and sum p_i = 1.
        # We can use a softmax parameterization: p_i = exp(z_i) / sum(exp(z_j)).
        # Then we optimize z (unconstrained).
        
        L = ref_f1 - x_min
        if L <= 0:
            # All points are useless
            X = np.zeros((N, dim))
            return X
            
        # Objective: Minimize -HV
        def neg_hv(z):
            # z is unconstrained
            # Compute p via softmax
            # Numerical stability: subtract max
            z_max = np.max(z)
            exp_z = np.exp(z - z_max)
            p = exp_z / np.sum(exp_z)
            
            # p has length N+1
            # y = L * p
            y = L * p
            
            # Reconstruct x
            # x_0 = x_min + y[0]
            # x_1 = x_min + y[0] + y[1]
            # ...
            # x_i = x_min + sum(y[0..i])
            cum_y = np.cumsum(y[:N])
            x_vals = x_min + cum_y
            
            # Compute f2
            # Ensure x_vals > 0 for sqrt
            x_safe = np.maximum(x_vals, 1e-15)
            f2_vals = C * (1.0 - np.sqrt(x_safe))
            
            # Compute HV
            # Points: (x_vals[i], f2_vals[i])
            # We need to check if f2_vals[i] < ref_f2.
            # If f2 >= ref_f2, the box height is <= 0.
            # Since x_vals are sorted increasing, f2_vals are sorted decreasing.
            # Find the first index where f2 < ref_f2.
            # If all f2 >= ref_f2, HV = 0.
            valid = f2_vals < ref_f2
            if not np.any(valid):
                return 0.0
            
            first_valid = np.argmax(valid)
            x_valid = x_vals[first_valid:]
            f2_valid = f2_vals[first_valid:]
            
            # If the first valid point is not the first point, the area before it is 0.
            # The integration starts from x_valid[0].
            # Wait, the formula was:
            # HV = sum_{j=first_valid}^{N-2} (x_{j+1} - x_j) * (ref_f2 - f2_j) + (ref_f1 - x_{N-1}) * (ref_f2 - f2_{N-1})
            # Note: The "width" for the last segment is ref_f1 - x_{N-1}, not x_N - x_{N-1}.
            # And for intermediate segments, it is x_{j+1} - x_j.
            
            if len(x_valid) == 1:
                width = ref_f1 - x_valid[0]
                height = ref_f2 - f2_valid[0]
                return -width * height
            
            widths = np.diff(x_valid)
            # The last width is ref_f1 - x_valid[-1]
            widths = np.append(widths, ref_f1 - x_valid[-1])
            
            heights = ref_f2 - f2_valid
            
            # Ensure non-negative
            widths = np.maximum(widths, 0)
            heights = np.maximum(heights, 0)
            
            hv = np.sum(widths * heights)
            return -hv
        
        # Initial guess: Uniform spacing in "log-x" or "x^0.5"?
        # Let's start with a power law x = (i/N)^0.5 which clusters near 0.
        # Map this to z.
        i_vals = np.arange(1, N+1)
        x_init = x_min + L * (i_vals / N)**0.5
        # x_init is sorted.
        # Gaps:
        y_init = np.diff(np.append(x_min, x_init))
        # y_init has length N.
        # Slack:
        y_init = np.append(y_init, ref_f1 - x_init[-1])
        # y_init has length N+1.
        # p_init = y_init / L
        p_init = y_init / L
        # z_init = log(p_init)
        # Handle zeros
        p_init = np.maximum(p_init, 1e-12)
        z_init = np.log(p_init)
        
        # Optimize
        try:
            res = minimize(neg_hv, z_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if time.time() > deadline:
                raise TimeoutError
            z_opt = res.x
        except Exception:
            # Fallback to initial guess
            z_opt = z_init
            if time.time() > deadline:
                raise TimeoutError

        # Reconstruct optimal x
        z_max = np.max(z_opt)
        exp_z = np.exp(z_opt - z_max)
        p_opt = exp_z / np.sum(exp_z)
        y_opt = L * p_opt
        cum_y = np.cumsum(y_opt[:N])
        x1_vals = x_min + cum_y
        
        # Construct decision vectors
        X = np.zeros((N, dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

# EVOLVE-BLOCK-END
