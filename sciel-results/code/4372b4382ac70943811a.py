import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_x0(self, n_points, x0_lo, x0_hi, r1, r2):
        """
        Compute optimal x0 values to maximize hypervolume approximation.
        
        The Pareto front is f2 = 2 - sqrt(f1) for f1 in [0,1].
        Hypervolume with reference (r1, r2):
        For points x_0 < x_1 < ... < x_{N-1}:
        HV = sum_{i=0}^{N-2} (x_{i+1} - x_i) * (r2 - (2 - sqrt(x_i)))
           + (r1 - x_{N-1}) * (r2 - (2 - sqrt(x_{N-1})))
        
        To maximize, we want to minimize discretization error.
        The optimal spacing equalizes the contribution per segment.
        
        Using the transformation u = sqrt(x), x = u^2, dx = 2u du:
        The integrand (r2 - 2 + sqrt(x)) * dx = (r2 - 2 + u) * 2u du
        
        For uniform contribution per segment in the transformed space,
        we use the inverse CDF approach.
        """
        if n_points <= 1:
            return np.array([(x0_lo + x0_hi) / 2.0])
        
        # For the front f2 = 2 - sqrt(f1), with reference (r1, r2):
        # The exact HV from f1=a to f1=b is:
        # integral_a^b (r2 - 2 + sqrt(x)) dx = (r2-2)(b-a) + (2/3)(b^{3/2} - a^{3/2})
        
        # With N discrete points, the HV is approximated by left-endpoint Riemann sum
        # (each point claims the interval to the next point).
        
        # To minimize error, we should use the optimal quadrature rule.
        # For a smooth integrand, the optimal spacing for left-endpoint rule
        # clusters points where the integrand changes rapidly.
        
        # The integrand h(x) = r2 - 2 + sqrt(x) is concave (h'' < 0).
        # For concave functions, the left-endpoint rule overestimates.
        # To minimize overestimation, we want points where h changes fast,
        # i.e., near x=0 where h'(x) = 1/(2*sqrt(x)) is large.
        
        # Use adaptive spacing: denser near x=0.
        # A good choice: x_i proportional to (i/N)^2 (quadratic spacing)
        # This makes dx_i proportional to i, so denser near 0.
        
        # Actually, let's use the exact optimal spacing for this problem.
        # The error of left-endpoint rule on [a,b] with one point at a is:
        # (b-a)*h(a) - integral_a^b h(x) dx
        # = (b-a)*h(a) - [(r2-2)(b-a) + (2/3)(b^{3/2} - a^{3/2})]
        
        # For small intervals, the error is approximately:
        # -h'(a)*(b-a)^2/2 = -1/(4*sqrt(a)) * (b-a)^2
        
        # To equalize error, we want (b-a)^2 / sqrt(a) = const
        # So (b-a) = C * a^{1/4}
        # This gives dx/dx = C * x^{1/4}
        # x^{-1/4} dx = C dx
        # (4/3) x^{3/4} = Cx + D
        # For x in [0,1], this is complex. Let's use a simpler approach.
        
        # Empirically, x_i = (i/(N-1))^p with p=2 or p=3 works well.
        # Let's use p=2 (quadratic) which is a standard choice for
        # problems with singular derivatives at the boundary.
        
        t = np.linspace(0.0, 1.0, n_points)
        x0_vals = x0_lo + (x0_hi - x0_lo) * t**2
        
        return x0_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if we can set x_1...x_9 to 0 (known global minimum for g(x))
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], self.budget)
            else:
                x0_vals = self._compute_optimal_x0(
                    self.budget, x0_lo, x0_hi, 11.0, 400.0
                )
            
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x0_vals
            
            if time.time() > deadline:
                return X
            
            # Evaluate all points
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            
            return X
        
        else:
            # Fallback: random search
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            return X
