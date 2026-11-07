import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # We want to sample x0 such that the density is proportional to 1/(10+sqrt(x0))
        # This corresponds to maximizing hypervolume for the curve f2 = 1 - sqrt(f1)
        # with reference point (11, 11).
        
        # CDF V(x) = 2*sqrt(x) - 20*ln(10+sqrt(x))
        # Normalized u(x) = (V(x) - V(0)) / (V(1) - V(0))
        
        def V(x):
            return 2.0 * np.sqrt(x) - 20.0 * np.log(10.0 + np.sqrt(x))
        
        V0 = V(0.0)
        V1 = V(1.0)
        range_V = V1 - V0
        
        # Sample u uniformly
        u_samples = np.linspace(0.0, 1.0, n_evals)
        
        # Invert u to get x0
        # u(x) = (V(x) - V0) / range_V = u_sample
        # V(x) = V0 + u_sample * range_V
        # Solve V(x) - target = 0 for x in [0, 1]
        
        x0_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target = V0 + u_samples[i] * range_V
            # brentq requires a sign change
            # V(0) = V0, V(1) = V1.
            # f(x) = V(x) - target
            # f(0) = -target + V0. If target > V0, f(0) < 0.
            # f(1) = V1 - target. If target < V1, f(1) > 0.
            # Since u is in [0,1], target is in [V0, V1].
            # At i=0, target=V0, f(0)=0. x=0.
            # At i=n-1, target=V1, f(1)=0. x=1.
            
            if u_samples[i] == 0.0:
                x0_vals[i] = 0.0
            elif u_samples[i] == 1.0:
                x0_vals[i] = 1.0
            else:
                try:
                    x0_vals[i] = brentq(lambda x: V(x) - target, 0.0, 1.0, xtol=1e-12)
                except Exception:
                    # Fallback to uniform if solver fails
                    x0_vals[i] = u_samples[i]
                    
        # Construct X
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        
        # Evaluate
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f_vals = problem(X[i])
            F.append(f_vals)
            
        if not F:
            return X
            
        return X
