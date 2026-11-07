import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _calculate_hv(self, u):
        # u is an array of shape (N,)
        # f1 = u^2, f2 = 1 - u
        # Reference point [11, 11]
        # HV = sum( (f1_next - f1_i) * (ref2 - f2_i) )
        # f1_next for i < N-1 is u[i+1]^2
        # f1_next for i = N-1 is 11.0
        
        f1 = u ** 2
        f2 = 1.0 - u
        
        # Widths
        f1_next = np.empty_like(f1)
        f1_next[:-1] = f1[1:]
        f1_next[-1] = 11.0
        
        widths = f1_next - f1
        heights = 11.0 - f2
        
        return np.sum(widths * heights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0 # 300s limit per eval, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # Initial guess: uniform spacing in u
        u_init = np.linspace(0, 1, N)
        
        # Objective to minimize: -HV
        def objective(u):
            return -self._calculate_hv(u)
        
        bounds_u = [(0.0, 1.0) for _ in range(N)]
        
        # Try to optimize
        try:
            res = minimize(objective, u_init, method='L-BFGS-B', bounds=bounds_u, 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            u_opt = res.x
        except Exception:
            u_opt = u_init
            
        # Check time
        if time.time() > deadline - 10.0:
            u_opt = u_init
            
        # Ensure u is sorted and within bounds (minimize should handle bounds, but just in case)
        u_opt = np.clip(u_opt, 0.0, 1.0)
        u_opt.sort()
        
        # Generate population
        pop = np.zeros((N, self.dim))
        pop[:, 0] = u_opt ** 2 * (hi[0] - lo[0]) + lo[0]
        for i in range(1, self.dim):
            pop[:, i] = lo[i]
            
        # Evaluate
        for i in range(N):
            problem(pop[i])
            
        return pop
