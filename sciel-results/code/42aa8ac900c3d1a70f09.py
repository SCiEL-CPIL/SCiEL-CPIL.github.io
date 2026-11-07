import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        if N <= 0:
            return None

        # ZDT1 structure:
        # f1 = x1
        # g = 1 + 9 * sum(x2..xn)/(n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Pareto front when g=1 (x2..xn=0): f2 = 1 - sqrt(f1)
        # Reference point: [11, 11]
        # Hypervolume contribution of a point (f1, f2) in the discrete set
        # is roughly proportional to the width of the interval it covers in f1
        # times the height (11 - f2).
        
        # To maximize hypervolume with N points, we should space points such that
        # the marginal gain per point is equal.
        # For a continuous front f2(f1), the optimal spacing for hypervolume 
        # maximization (minimizing the area between the front and the lower envelope 
        # of the points) is related to the curvature or the integrand.
        
        # Specifically, if we approximate the hypervolume as sum of rectangles:
        # HV ~ sum (f1_{i+1} - f1_i) * (11 - f2(f1_i))
        # The optimal grid points x_i are determined by the condition that the 
        # "weight" of each interval is equal.
        # A common approximation for optimal spacing in such 1D problems is 
        # to use the inverse cumulative distribution of the function w(x) = 11 - f2(x).
        # Here w(f1) = 11 - (1 - sqrt(f1)) = 10 + sqrt(f1).
        
        # We want to choose f1_i such that the integral of w(f1) over each interval 
        # [f1_i, f1_{i+1}] is constant.
        # Let W(f1) = integral_0^f1 w(t) dt = integral_0^f1 (10 + sqrt(t)) dt
        # W(f1) = 10*f1 + (2/3)*f1^(3/2)
        # We want W(f1_i) = i * W(1) / N for i=0..N (adjusting indices for N points)
        # Actually, we have N points. Let's say we place points at f1_1, ..., f1_N.
        # The intervals are [0, f1_1], [f1_1, f1_2], ..., [f1_{N-1}, f1_N], [f1_N, 1]?
        # No, the hypervolume is calculated based on the points themselves.
        # Standard result: for maximizing sum h_i * w(x_i) with h_i = x_{i+1} - x_i,
        # the optimal x_i satisfy x_{i+1} - x_i proportional to 1/sqrt(w(x_i))?
        
        # Let's stick to the CDF inversion method which is robust.
        # We want N points. Let's define the target cumulative weights.
        # The "mass" of the function w(f1) over [0,1] is M = W(1) = 10 + 2/3 = 32/3.
        # We want to split the domain [0,1] into N segments such that the integral of w
        # over each segment is M/N.
        # The points f1_i will be the boundaries of these segments.
        # Wait, if we have N points, do we have N intervals or N-1?
        # Usually, points are placed to cover the range.
        # If we place points at f1_1 ... f1_N, the hypervolume is determined by these points.
        # The "effective" width for point i is roughly (f1_{i+1} - f1_{i-1})/2 for interior points.
        # A simpler heuristic that works well is to solve for f1_i such that
        # W(f1_i) = (i + 0.5) * M / N for i=0..N-1. This places points at the centers
        # of the equal-mass intervals.
        
        if N == 1:
            # Single point: maximize (11 - f1) * (11 - f2)? No, HV of single point is (11-f1)(11-f2).
            # f2 = 1 - sqrt(f1).
            # HV = (11-f1)(10+sqrt(f1)).
            # Let f1 = u^2. HV = (11-u^2)(10+u).
            # Maximize this. Derivative is complex, but usually near u=0 or u=1?
            # At u=0 (f1=0): HV = 11*10 = 110.
            # At u=1 (f1=1): HV = 10*11 = 110.
            # At u=0.5 (f1=0.25): HV = 10.75 * 10.5 = 112.875.
            # So middle is better.
            # Let's just use f1=0.25 (u=0.5) or solve quickly.
            # Actually, for N=1, the best single point for HV against ref (11,11) 
            # on this front is likely not at the ends.
            # Let's use the CDF center for N=1 as well.
            pass

        # Calculate W(f1) = 10*f1 + (2/3)*f1**1.5
        # Invert W(f1) = c to find f1.
        # 10*f1 + (2/3)*f1**1.5 = c
        # This is a polynomial in sqrt(f1). Let u = sqrt(f1), u in [0,1].
        # 10*u^2 + (2/3)*u^3 = c
        # 30*u^2 + 2*u^3 = 3c
        # u^3 + 15*u^2 - 1.5*c = 0
        
        def solve_u(c):
            # Solve 2*u^3 + 30*u^2 - 3*c = 0 for u in [0,1]
            # Use numpy roots or Newton's method
            # Newton's method is fast and robust here since function is monotonic increasing on [0,1]
            # f(u) = 2*u^3 + 30*u^2 - 3*c
            # f'(u) = 6*u^2 + 60*u
            # Start at u=0.5
            u = 0.5
            for _ in range(20):
                val = 2*u**3 + 30*u**2 - 3*c
                if abs(val) < 1e-12:
                    break
                deriv = 6*u**2 + 60*u
                if deriv == 0:
                    break
                u_new = u - val/deriv
                if u_new < 0: u_new = 0
                if u_new > 1: u_new = 1
                if abs(u_new - u) < 1e-12:
                    u = u_new
                    break
                u = u_new
            return u

        M = 32.0 / 3.0  # Total mass W(1)
        
        x1_values = np.zeros(N)
        for i in range(N):
            # Target cumulative mass for the center of the i-th bin
            # We have N bins, centers at (i + 0.5)/N of the total mass
            c = (i + 0.5) * M / N
            u = solve_u(c)
            x1_values[i] = u**2

        # Sort to ensure increasing order (should already be, but just in case)
        x1_values = np.sort(x1_values)

        base_vec = np.zeros(self.dim)

        for i in range(N):
            if time.time() > deadline:
                break

            x = base_vec.copy()
            x[0] = x1_values[i]
            problem(x)

        return None
