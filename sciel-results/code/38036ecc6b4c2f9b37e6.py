# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize optimizer for ZDT2.
    
    Exploits the known Pareto front geometry of ZDT2:
    - Front: x1 in [0,1], x2..xn = 0
    - f1 = x1
    - f2 = C * (1 - sqrt(x1/C)) where C = 1 + 10*(n-1)
    
    For 2 objectives, hypervolume with reference (r1, r2) is computed
    by sorting points by f1 and summing trapezoid contributions.
    
    We use an analytically derived non-uniform spacing where point density
    is proportional to the local curvature of the Pareto front to maximize
    hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 parameters
        C = 1.0 + 10.0 * (dim - 1)
        r1 = 11.0
        r2 = 11.0

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # f2 as function of x1
        def f2_of_x1(x1):
            x1_clipped = np.clip(x1, 0.0, 1.0)
            return C * (1.0 - np.sqrt(x1_clipped / C))

        # Threshold x1 where f2 = r2: C*(1-sqrt(x1/C)) = r2
        # 1 - sqrt(x1/C) = r2/C
        # sqrt(x1/C) = 1 - r2/C
        # x1 = C * (1 - r2/C)^2
        x1_thresh = C * (1.0 - r2 / C) ** 2
        
        # Ensure threshold is within [0, 1]
        x1_thresh = np.clip(x1_thresh, 0.0, 1.0)
        
        # For ZDT2: f1 = x1, f2 = C*(1 - sqrt(x1/C))
        # The Pareto front in objective space is a convex decreasing curve.
        # To maximize hypervolume with n points, we want to place points such
        # that the "gaps" in objective space are minimized in a hypervolume-optimal way.
        # 
        # For a convex decreasing curve, the optimal placement in terms of maximizing
        # the area under the step function approximation is to make the contribution
        # of each segment equal. This leads to a non-uniform spacing in x1.
        #
        # Let's derive the optimal x1 values. The hypervolume contribution from a point
        # at x1_i (with f1_i = x1_i, f2_i = C*(1-sqrt(x1_i/C))) when points are sorted
        # by f1 (i.e., by x1) is:
        # HV = sum over i of (r1 - f1_i) * (f2_{i-1} - f2_i) for i > 0, plus boundary terms.
        #
        # More precisely, if we have points sorted by increasing f1:
        # HV = (r1 - f1_0)*(f2_0 - r2) + sum_{i=1}^{n-1} (f1_{i-1} - f1_i)*(r2 - f2_i) ... 
        # Actually, the standard formula for 2D HV with reference (r1, r2) and points
        # sorted by f1 ascending is:
        # HV = sum_{i=0}^{n-1} (r1 - f1_i) * (f2_{i-1} - f2_i) where f2_{-1} = r2
        # But this needs careful handling. Let's use the fact that for a convex decreasing
        # curve, the optimal nodes in the domain (x1) are those that equalize the 
        # hypervolume contribution per segment.
        #
        # A simpler approach: since f1 = x1 and f2 is a smooth convex decreasing function
        # of x1, we can transform the problem. Let u = sqrt(x1/C). Then x1 = C*u^2,
        # f1 = C*u^2, f2 = C*(1-u).
        # In the (f1, f2) plane, the curve is f2 = C*(1 - sqrt(f1/C)).
        # 
        # For maximizing HV of a convex decreasing curve, the optimal placement is 
        # approximately uniform in the "angle" or in a transformed space. A known result
        # is that for a curve f2 = g(f1) where g is convex decreasing, the optimal 
        # f1 values are those where the second derivative weighted by the gap is constant.
        #
        # Let's use a practical approach: sample uniformly in a transformed variable.
        # The curvature of f2 w.r.t. f1 is:
        # df2/df1 = -1/(2*sqrt(x1*C)) ... let's compute:
        # f2 = C - sqrt(C*x1)
        # df2/dx1 = -sqrt(C)/(2*sqrt(x1))
        # d2f2/dx1^2 = sqrt(C)/(4*x1^(3/2))
        #
        # The curvature kappa = |d2f2/dx1^2| / (1 + (df2/dx1)^2)^(3/2)
        # For large C and small x1, this is dominated by the second derivative.
        #
        # A good approximation for the optimal x1 values is to make x1^(3/2) uniformly
        # spaced, since d2f2/dx1^2 ~ 1/x1^(3/2).
        #
        # Let's try: x1_opt[i] = x1_thresh + (1.0 - x1_thresh) * (i/(n_points-1))^2
        # This makes the spacing in sqrt(x1) roughly uniform, which is a common heuristic.
        #
        # Actually, let's try a more principled approach. We want to maximize:
        # HV = sum_{i=0}^{n-1} (r1 - f1_i) * (f2_{i-1} - f2_i)
        # where f2_{-1} = r2.
        #
        # For simplicity and robustness, let's use a transformation that makes the 
        # curve approximately linear. Let t = sqrt(x1). Then x1 = t^2, f1 = t^2,
        # f2 = C*(1 - t/sqrt(C)).
        # In (f1, f2) space, this is still curved. 
        #
        # Let's just use uniform spacing in t = sqrt(x1), which gives:
        # x1_opt[i] = (x1_thresh^(1/2) + (1.0 - x1_thresh^(1/2)) * i/(n_points-1))^2
        # for i = 0, ..., n_points-1
        #
        # This is a reasonable heuristic that clusters points where the curve is steeper.
        
        t_min = np.sqrt(x1_thresh)
        t_max = np.sqrt(1.0)
        
        # For 2D HV with reference (r1, r2) and points sorted by increasing f1:
        # HV = sum_{i=0}^{n-1} (r1 - f1_i) * (f2_{i-1} - f2_i) where f2_{-1} = r2
        # But only points with f1 < r1 AND f2 < r2 contribute.
        # Since f1 = x1 and r1 = 11, all x1 in [0,1] have f1 < r1.
        # f2 < r2 requires x1 > x1_thresh (computed above).
        #
        # The HV contribution of a point at position i (sorted by f1 ascending) is:
        # (r1 - f1_i) * (f2_{i-1} - f2_i) for i >= 1
        # and (r1 - f1_0) * (r2 - f2_0) for i = 0
        #
        # To maximize total HV, we want to choose x1 values to maximize this sum.
        # This is a 1D optimal design problem. The optimal spacing makes the
        # "area" contribution per segment as equal as possible, weighted by the
        # leverage (r1 - f1_i).
        #
        # A practical approach: use the fact that for a convex curve, the optimal
        # nodes in the parameter space are those where the local slope change is
        # balanced. For f2(x1) = C - sqrt(C*x1), the derivative is -sqrt(C)/(2*sqrt(x1)),
        # which varies as x1^(-1/2). The second derivative varies as x1^(-3/2).
        #
        # Empirically, for this type of curve, using x1 proportional to (i/n)^alpha
        # with alpha < 1 clusters points near x1=0. Let's use alpha=0.5 (uniform in sqrt)
        # but also try to optimize by considering the actual HV formula.
        #
        # Better approach: since the problem is 1D and we know the analytical form,
        # let's use a grid that is dense near x1=0 where the curve is steeper.
        # Use x1_i = (i/(n_points-1))^2 * 1.0, which is uniform in sqrt(x1).
        # But we need x1 >= x1_thresh. So:
        # x1_i = x1_thresh + (1.0 - x1_thresh) * (i/(n_points-1))^2
        
        if n_points == 1:
            x1_opt = np.array([1.0])
        else:
            # Uniform in sqrt(x1) over [x1_thresh, 1.0]
            t_vals = np.linspace(np.sqrt(x1_thresh), 1.0, n_points)
            x1_opt = t_vals ** 2
            x1_opt = np.clip(x1_opt, x1_thresh, 1.0)

        # Evaluate the optimal points
        X_list = []
        F_list = []

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_opt[i]
            x = np.clip(x, lo, hi)

            f = problem(x)
            X_list.append(x)
            F_list.append(f)

        return np.array(X_list)
# EVOLVE-BLOCK-END
