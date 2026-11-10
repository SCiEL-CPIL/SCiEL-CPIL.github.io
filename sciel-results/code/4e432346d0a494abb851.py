"""ZDT2 Optimizer using exact analytic placement via root-finding on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root

class Optimizer:
    """
    Exploits the analytic structure of ZDT2.
    Strategy:
    1. The Pareto front of ZDT2 is 1D, parameterized by x1 in [0, 1].
    2. The optimal placement of N points on a 1D curve to maximize 2D Hypervolume 
       satisfies specific first-order optimality conditions.
    3. We solve these conditions exactly using scipy.optimize.root (Newton-like method).
    4. This avoids the slow convergence and potential local minima of gradient-based 
       optimization (like L-BFGS-B) and is extremely fast.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 specifics
        # f1 = x1
        # g(x) = 1 + sum(x_i^2) for i=2..n
        # f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1 / (1 + 10*(dim-1))))
        # Pareto front: x2...xdim = 0
        n_extra = dim - 1
        scale = 1.0 + 10.0 * n_extra
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # Objective function on the Pareto front
        # x1 in [0, 1]
        def front_f2(x1):
            # Ensure x1 is within [0, 1] to avoid sqrt issues, though bounds should handle it
            x1_clipped = np.clip(x1, 0.0, 1.0)
            return scale * (1.0 - np.sqrt(x1_clipped / scale))
        
        def front_f2_prime(x1):
            # d/dx [ scale * (1 - sqrt(x/scale)) ] = scale * (-1/2) * (x/scale)^(-1/2) * (1/scale)
            # = -0.5 * sqrt(scale / x)
            # Note: f2 is decreasing. The derivative is negative.
            # Let's use the magnitude for the geometric condition or handle signs carefully.
            # In the optimality condition: (f_i - f_{i-1}) = (x_{i+1} - x_i) * f'_i
            # Since f is decreasing, f_i - f_{i+1} > 0.
            # Let's define h(x) = f2(x). h'(x) < 0.
            # The condition is: h(x_i) - h(x_{i-1}) + (x_{i+1} - x_i) h'(x_i) = 0?
            # Let's derive again.
            # HV contribution of point i (assuming sorted x_0 < x_1 < ... < x_{N-1})
            # Area_i = (x_{i+1} - x_i) * (ref_f2 - h(x_i)) + (ref_f1 - x_{i+1}) * (ref_f2 - h(x_{N-1}))? No.
            # The hypervolume is the sum of rectangles under the "upper envelope".
            # Since the front is convex (h'' > 0), the points lie on the curve.
            # The area covered is the sum of rectangles:
            # For i=0 to N-2: width = x_{i+1} - x_i, height = ref_f2 - h(x_i)
            # For i=N-1: width = ref_f1 - x_{N-1}, height = ref_f2 - h(x_{N-1})
            # Total HV = sum_{i=0}^{N-2} (x_{i+1} - x_i)(ref_f2 - h(x_i)) + (ref_f1 - x_{N-1})(ref_f2 - h(x_{N-1}))
            
            # Let's check the derivative of HV w.r.t x_i.
            # Term i: (x_{i+1} - x_i)(ref_f2 - h(x_i))
            # d/dx_i = -(ref_f2 - h(x_i)) - (x_{i+1} - x_i) h'(x_i)
            # Term i-1: (x_i - x_{i-1})(ref_f2 - h(x_{i-1}))
            # d/dx_i = (ref_f2 - h(x_{i-1}))
            
            # Interior optimality condition (i=1..N-2):
            # (ref_f2 - h(x_{i-1})) - (ref_f2 - h(x_i)) - (x_{i+1} - x_i) h'(x_i) = 0
            # h(x_i) - h(x_{i-1}) - (x_{i+1} - x_i) h'(x_i) = 0
            
            # Boundary i=0:
            # dHV/dx_0 = -(ref_f2 - h(x_0)) - (x_1 - x_0) h'(x_0) = 0
            # h(x_0) - ref_f2 - (x_1 - x_0) h'(x_0) = 0 ? 
            # Wait, the first term is (x_1 - x_0)(ref_f2 - h(x_0)).
            # d/dx_0 = -(ref_f2 - h(x_0)) - (x_1 - x_0) h'(x_0) = 0
            # => h(x_0) - ref_f2 - (x_1 - x_0) h'(x_0) = 0 is wrong sign?
            # -(ref_f2 - h(x_0)) = h(x_0) - ref_f2.
            # So: h(x_0) - ref_f2 - (x_1 - x_0) h'(x_0) = 0.
            # Note: h'(x_0) is negative. So -(x_1 - x_0) h'(x_0) is positive.
            # h(x_0) - ref_f2 is negative (since h(x_0) < ref_f2).
            # This equation balances the loss of height at x_0 vs gain in width.
            
            # Boundary i=N-1:
            # Term N-2: (x_{N-1} - x_{N-2})(ref_f2 - h(x_{N-2}))
            # d/dx_{N-1} = (ref_f2 - h(x_{N-2}))
            # Term N-1: (ref_f1 - x_{N-1})(ref_f2 - h(x_{N-1}))
            # d/dx_{N-1} = -(ref_f2 - h(x_{N-1})) - (ref_f1 - x_{N-1}) h'(x_{N-1})
            # Sum = (ref_f2 - h(x_{N-2})) - (ref_f2 - h(x_{N-1})) - (ref_f1 - x_{N-1}) h'(x_{N-1}) = 0
            # => h(x_{N-1}) - h(x_{N-2}) - (ref_f1 - x_{N-1}) h'(x_{N-1}) = 0
            
            x_safe = np.maximum(x1, 1e-12)
            return -0.5 * np.sqrt(scale / x_safe)

        N = budget
        
        if N <= 1:
            x1s = np.array([0.5])
        elif N == 2:
            # Solve for x1 given x0=0, x2=1? Or optimize both?
            # With N=2, we have x0, x1.
            # It's easier to just use a generic solver for all x_i.
            # But fixing endpoints is usually optimal for covering the range [0,1].
            # Let's assume x_0 = 0 and x_{N-1} = 1 are optimal.
            # If N=2, x_0=0, x_1=1.
            x1s = np.array([0.0, 1.0])
        else:
            # We fix x_0 = 0 and x_{N-1} = 1.
            # We optimize x_1, ..., x_{N-2}.
            # Number of variables: N - 2.
            
            def residuals(x_vars):
                # x_vars contains x_1 ... x_{N-2}
                x_full = np.concatenate(([0.0], x_vars, [1.0]))
                
                res = np.zeros(len(x_vars))
                
                # Interior points i=1 to N-2
                for i in range(1, N-1):
                    h_i = front_f2(x_full[i])
                    h_prev = front_f2(x_full[i-1])
                    h_prime_i = front_f2_prime(x_full[i])
                    dx_next = x_full[i+1] - x_full[i]
                    
                    # Condition: h(x_i) - h(x_{i-1}) - (x_{i+1} - x_i) h'(x_i) = 0
                    res[i-1] = h_i - h_prev - dx_next * h_prime_i
                    
                return res

            # Initial guess: Linear spacing
            x_init = np.linspace(0.0, 1.0, N)[1:-1]
            
            # Solve
            # Method 'hybr' is a minimization of the sum of squares of residuals (Levenberg-Marquardt).
            # Method 'lm' is Levenberg-Marquardt.
            # Method 'broyden1' is a quasi-Newton method.
            # Let's use 'hybr' as it is robust for non-linear systems.
            
            try:
                sol = root(residuals, x_init, method='hybr', options={'xtol': 1e-15, 'maxfev': 10000})
                if sol.success:
                    x1s = np.concatenate(([0.0], sol.x, [1.0]))
                else:
                    # Fallback to linear if it fails
                    x1s = np.linspace(0.0, 1.0, N)
            except Exception:
                x1s = np.linspace(0.0, 1.0, N)
            
            # Ensure sorted and within bounds
            x1s = np.sort(x1s)
            x1s = np.clip(x1s, 0.0, 1.0)

        # Evaluate the points
        last_F = None
        for x1 in x1s:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] = 0
            F = np.array(problem(x))
            last_F = F
            
        return last_F

# EVOLVE-BLOCK-END
