import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Linear Slope function.
    Uses a linear model fit to identify the global minimum at a box vertex,
    followed by a local search to handle potential non-linearities or numerical issues.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # 1. Evaluate at origin
        x0 = np.zeros(self.dim)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, np.inf
        best_f = f0
        best_x = x0.copy()

        # 2. Evaluate at basis vectors to estimate gradient
        w = np.zeros(self.dim)
        for i in range(self.dim):
            if time.time() > deadline or calls_made >= self.budget:
                break
            xi = np.zeros(self.dim)
            xi[i] = 1.0
            fi = safe_call(xi)
            if fi is None:
                break
            w[i] = fi - f0
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # 3. Determine optimal vertex based on linear model
        # f(x) = w^T x + b. Minimize by setting x_i = -5 if w_i > 0, else 5.
        x_opt = np.zeros(self.dim)
        for i in range(self.dim):
            if w[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
        
        f_opt = safe_call(x_opt)
        if f_opt is not None and f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()

        # 4. Local search with (1+1)-ES for remaining budget
        # This helps if the function is not perfectly linear or if numerical precision matters.
        if calls_made < self.budget and time.time() < deadline:
            sigma = 0.5
            rng = np.random.default_rng(12345)
            # Start from the best point found so far
            current_x = best_x.copy()
            current_f = best_f
            
            while calls_made < self.budget and time.time() < deadline:
                y = current_x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < current_f:
                    current_x = y
                    current_f = fy
                    best_x = current_x
                    best_f = current_f
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        return best_x, best_f
