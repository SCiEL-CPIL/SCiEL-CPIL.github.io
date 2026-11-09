import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        dim = self.dim
        budget = self.budget
        
        # Storage for evaluated points
        self.X_evals = []
        self.Y_evals = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            # Clip to bounds just in case, though func does it
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            self.X_evals.append(x.copy())
            self.Y_evals.append(fx)
            return fx

        # Phase 1: Initial Exploration
        # Use a small number of points to find a good starting basin
        # Mix of random and origin-centered
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Generate initial points
        init_pts = []
        # 5 points near origin
        for _ in range(5):
            init_pts.append(rng.uniform(-1.0, 1.0, size=dim))
        # 15 points random in box
        for _ in range(15):
            init_pts.append(rng.uniform(-5.0, 5.0, size=dim))
        
        for x in init_pts:
            if self.call_count >= budget or time.time() > deadline - 10:
                break
            wrapped_func(x)
            
        if self.best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: (1+1)-ES with 1/5th Success Rule
        # This is the core strategy requested
        x_curr = self.best_x.copy()
        f_curr = self.best_f
        sigma = 1.0  # Initial step size
        
        # Parameters for 1/5th rule
        window_size = 10  # Number of steps to evaluate success rate
        successes = 0
        steps_in_window = 0
        c_inc = 1.2  # Increase factor
        c_dec = 0.8  # Decrease factor
        
        # Budget allocation: Use 80% for ES, 20% for Polish
        es_budget = int(budget * 0.8)
        max_es_steps = es_budget - self.call_count
        
        for step in range(max_es_steps):
            if self.call_count >= budget or time.time() > deadline - 20:
                break
            
            # Generate candidate
            # Standard normal perturbation
            perturbation = rng.standard_normal(dim) * sigma
            x_new = x_curr + perturbation
            
            # Evaluate
            f_new = wrapped_func(x_new)
            
            # Update 1/5th rule statistics
            if f_new < f_curr:
                x_curr = x_new
                f_curr = f_new
                successes += 1
            
            steps_in_window += 1
            
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 0.2:
                    sigma *= c_inc
                else:
                    sigma *= c_dec
                # Reset window
                successes = 0
                steps_in_window = 0
                
            # Safety: Don't let sigma get too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 5.0:
                sigma = 5.0

        # Phase 3: Nelder-Mead Polish
        # Use remaining budget for local refinement
        remaining_budget = budget - self.call_count
        if remaining_budget > 10 and time.time() < deadline - 5:
            def nm_func(x):
                if self.call_count >= budget or time.time() > deadline - 2:
                    # Return a large value to stop NM, or best_f?
                    # Returning best_f might confuse NM. 
                    # Better to raise an exception or return inf?
                    # Scipy handles exceptions by stopping.
                    raise ValueError("Budget or time exceeded")
                return wrapped_func(x)
            
            try:
                # Set maxiter to use up remaining budget
                # NM uses ~N+1 evals per iter
                maxiter_nm = max(1, remaining_budget // 6)
                
                res = minimize(
                    nm_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter_nm,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except ValueError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
