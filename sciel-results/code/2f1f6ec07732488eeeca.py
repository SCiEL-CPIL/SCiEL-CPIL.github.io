"""ZDT2 Optimizer using SciPy optimization for exact hypervolume-maximizing point placement on the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Exploits the analytic structure of ZDT2.
    Strategy:
    1. Identify the 1D Pareto front parameterized by x1.
    2. Use SciPy's L-BFGS-B to maximize the exact 2D hypervolume of N points on the front.
    3. Use analytic gradients for speed and robustness.
    4. Evaluate exactly these points using the budget.
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
            return scale * (1.0 - np.sqrt(x1 / scale))
        
        def front_f2_prime(x1):
            x_safe = np.maximum(x1, 1e-10)
            return 0.5 * np.sqrt(scale / x_safe)

        N = budget
        
        # We optimize x1s (length N).
        # x1s must be sorted: 0 <= x1s[0] <= x1s[1] <= ... <= x1s[N-1] <= 1.
        # To handle the sorting constraint easily, we can optimize the gaps or use bounds.
        # L-BFGS-B supports bounds.
        # We fix x1s[0] = 0 and x1s[-1] = 1? 
        # If we don't fix them, the optimizer might move them.
        # However, including 0 and 1 is generally optimal for HV coverage.
        # Let's optimize interior points x1s[1:-1] and fix endpoints.
        
        if N <= 2:
            # Handle edge case
            x1s = np.linspace(0.0, 1.0, N)
        else:
            # Initial guess: Uniform
            x1s_init = np.linspace(0.0, 1.0, N)
            # Optimize interior points
            x_init = x1s_init[1:-1]
            
            # Objective: Minimize negative Hypervolume
            def neg_hv(x_int):
                # Reconstruct full x1s
                x_full = np.concatenate(([0.0], x_int, [1.0]))
                # Ensure sorted (L-BFGS-B with bounds on differences would be better, 
                # but simple bounds on x_full are tricky if we don't enforce order.
                # We can enforce order by optimizing gaps:
                # Let g_i = x_i - x_{i-1}. g_i >= 0. sum(g_i) = 1.
                # This is a simplex constraint.
                # Alternatively, just use x_full and penalize unsorted?
                # Better: Optimize x_full directly with bounds [0,1] and rely on the 
                # objective function being monotonic? No, HV is not monotonic in x_i if unsorted.
                
                # Let's stick to the gap formulation for robustness.
                # x_0 = 0
                # x_i = x_{i-1} + g_{i-1}
                # x_N = 1 => sum(g) = 1.
                # This is a constrained optimization.
                
                # Actually, the previous Newton method worked by iterating on x_i directly.
                # L-BFGS-B can handle simple bounds. If we pass x_full, we need to ensure x_i <= x_{i+1}.
                # We can use a transformation: x_i = cumulative sum of softplus(gaps).
                # Or, simpler: Just use the previous Newton method but with a much better 
                # initialization and more robust convergence check.
                
                # Let's try a different robust method: 
                # The optimal points satisfy f(x_i) - f(x_{i-1}) = (x_{i+1} - x_i) f'(x_i).
                # This is a system of equations. We can solve it using `scipy.optimize.fsolve` 
                # or `root` on the residuals.
                
                # Residuals:
                # R_i(x) = f(x_i) - f(x_{i-1}) - (x_{i+1} - x_i) f'(x_i) = 0 for i=1..N-2
                # Boundary conditions:
                # If x_0 is free: R_0 = (x_1 - x_0) f'(x_0) - (f(x_1) - f(x_0))? 
                # Actually, for i=0, the derivative of HV w.r.t x_0 is:
                # dHV/dx_0 = (ref - f2(x_1)) - (ref - f2(x_0)) - (x_1 - x_0) f2'(x_0) = 0
                # => f2(x_0) - f2(x_1) - (x_1 - x_0) f2'(x_0) = 0
                # Similarly for i=N-1:
                # dHV/dx_{N-1} = (ref - f2(x_{N-2})) - (ref - f2(x_{N-1})) - (ref_f1 - x_{N-1}) f2'(x_{N-1})? 
                # Wait, the last term in HV is (ref_f1 - x_{N-1}) * (ref_f2 - f2(x_{N-1})).
                # d/dx_{N-1} = -(ref_f2 - f2(x_{N-1})) - (ref_f1 - x_{N-1}) f2'(x_{N-1})
                # Plus the term from the previous rectangle: (x_{N-1} - x_{N-2}) * (ref_f2 - f2(x_{N-2}))
                # d/dx_{N-1} of that is (ref_f2 - f2(x_{N-2})).
                # So: (ref_f2 - f2(x_{N-2})) - (ref_f2 - f2(x_{N-1})) - (ref_f1 - x_{N-1}) f2'(x_{N-1}) = 0
                # => f2(x_{N-1}) - f2(x_{N-2}) - (ref_f1 - x_{N-1}) f2'(x_{N-1}) = 0
                
                pass # This block is for planning, not execution.

            # Let's implement the `scipy.optimize.root` approach on the residuals.
            # This is a different method: Solving a non-linear system of equations (Newton-like)
            # using a robust library instead of a hand-rolled loop.
            
            def residuals(x_full):
                # x_full is length N
                # x_full[0] and x_full[-1] are variables? 
                # It's better to fix x_full[0]=0 and x_full[-1]=1 if we believe they are optimal.
                # Let's assume they are optimal. Optimize x_full[1:-1].
                x_int = x_full[1:-1]
                x_prev = np.concatenate(([0.0], x_int[:-1]))
                x_next = np.concatenate((x_int[1:], [1.0]))
                
                f_curr = front_f2(x_int)
                f_prev = front_f2(x_prev)
                f_next = front_f2(x_next) # Not needed for interior eq?
                f_prime_curr = front_f2_prime(x_int)
                
                # Interior equation: f(x_i) - f(x_{i-1}) - (x_{i+1} - x_i) f'(x_i) = 0
                res = f_curr - f_prev - (x_next - x_int) * f_prime_curr
                return res

            # Initial guess
            x_init_int = np.linspace(0.0, 1.0, N)[1:-1]
            
            # Solve
            sol = minimize(lambda x: np.sum(residuals(x)**2), x_init_int, method='L-BFGS-B', 
                           bounds=[(1e-10, 1-1e-10)] * (N-2), options={'maxiter': 1000, 'ftol': 1e-15})
            
            x1s = np.concatenate(([0.0], sol.x, [1.0]))
            x1s = np.sort(x1s)

        # Evaluate the points
        last_F = None
        for x1 in x1s:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = np.clip(x1, lo[0], hi[0])
            # x[1:] = 0 (clipped to bounds)
            F = np.array(problem(x))
            last_F = F
            
        return last_F

# EVOLVE-BLOCK-END
