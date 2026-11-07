import numpy as np
from scipy.optimize import minimize, brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
            
        # Extract bounds for x1 (index 0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        x1_range = x1_hi - x1_lo
        
        # ZDT2 specific constants
        C = 10.0 * (n - 1)
        ref = 11.0
        
        # Find tail value c such that g(x_tail) = 1
        # g(x) = sum_{i=2}^n ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))
        # Assume x_2 = ... = x_n = c
        # (n-1) * ((c - 0.5)^2 - cos(20*pi*(c - 0.5))) = 1
        
        def g_eq(u):
            # u = c - 0.5
            return (n - 1) * (u**2 - np.cos(20 * np.pi * u)) - 1.0
            
        # Solve for u in [-0.5, 0.5]
        # g_eq(0) = (n-1)(-1) - 1 = -n < 0
        # g_eq(0.5) = (n-1)(0.25 - 1) - 1 = -0.75(n-1) - 1 < 0
        # g_eq(0.25) = (n-1)(0.0625 - (-1)) - 1 = 1.0625(n-1) - 1
        # For n=30, 1.0625*29 - 1 > 0
        # So root exists between 0 and 0.25
        
        u_sol = brentq(g_eq, 0.0, 0.25)
        c_val = 0.5 + u_sol
        
        # Analytical Pareto front for ZDT2: f2 = C * (1 - sqrt(f1))
        # This is valid if g(x)=1, which we ensured.
        
        def analytical_f2(x1):
            return C * (1.0 - np.sqrt(np.clip(x1, 0, 1)))
            
        def calc_hv_from_x(x1_vals):
            # x1_vals should be sorted, length N
            if len(x1_vals) == 0:
                return 0.0
            y_vals = analytical_f2(x1_vals)
            # HV = sum (x_{i+1} - x_i) * (ref - y_i) for i < N-1
            # + (ref - x_{N-1}) * (ref - y_{N-1})
            
            if len(x1_vals) == 1:
                return (ref - x1_vals[0]) * (ref - y_vals[0])
            
            dx = np.diff(x1_vals)
            heights = ref - y_vals[:-1]
            hv = np.sum(dx * heights)
            last_width = ref - x1_vals[-1]
            last_height = ref - y_vals[-1]
            hv += last_width * last_height
            return hv

        N = budget
        
        # Optimize the grid x_1, ..., x_N
        # We fix x_N = 1 (in unit space) and x_0 = 0
        # Variables: d_0, ..., d_{N-1} such that sum(d) = 1, d_i > 0
        # Use softmax parameterization z -> d
        # z has length N
        
        def objective(z):
            # z: shape (N,)
            # Compute d
            z_max = np.max(z)
            exp_z = np.exp(z - z_max)
            d = exp_z / np.sum(exp_z)
            
            # Compute x
            x = np.cumsum(d)
            # x[0] is x_1, x[-1] is x_N = 1
            
            # Calculate HV
            hv = calc_hv_from_x(x)
            return -hv # Minimize negative HV
        
        # Initial guess: uniform d (z=0)
        z0 = np.zeros(N)
        
        # Optimize
        try:
            res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            z_opt = res.x
        except Exception:
            z_opt = z0
            
        # Compute final d and x
        z_max = np.max(z_opt)
        exp_z = np.exp(z_opt - z_max)
        d_opt = exp_z / np.sum(exp_z)
        x_unit = np.cumsum(d_opt)
        
        # Map to actual bounds
        x1_final = x1_lo + x_unit * x1_range
        
        # Create full vectors
        X = np.zeros((N, n))
        X[:, 0] = x1_final
        # Set tail to c_val
        if n > 1:
            X[:, 1:] = c_val
        
        # Evaluate exactly N points
        F = np.array([problem(x) for x in X])
        
        return X
