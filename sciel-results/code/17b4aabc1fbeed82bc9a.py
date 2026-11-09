"""Multi-objective optimizer for RE21 using Epsilon-Constraint Method with L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = [] # List of (x, f) tuples
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290.0 # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        def add_to_archive(x, f):
            # Remove points dominated by f
            new_archive = []
            for x_old, f_old in self.archive:
                # If f_old is dominated by f, discard f_old
                if np.all(f <= f_old) and np.any(f < f_old):
                    continue
                new_archive.append((x_old, f_old))
            new_archive.append((x, f))
            self.archive = new_archive

        # Phase 1: Initialization to find extreme points
        n_init = min(20, self.budget)
        X_init = np.random.random((n_init, self.dim))
        X_init = lo + span * X_init
        
        for x in X_init:
            if self.eval_count >= self.budget:
                break
            f = problem(x)
            self.eval_count += 1
            add_to_archive(x, f)
            
        # Find approximate optima for each objective individually
        # This helps establish the range of the Pareto front
        if self.eval_count < self.budget - 10:
            # Minimize f1
            def obj1(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                return f[0]
            
            res1 = minimize(obj1, np.random.uniform(lo, hi), method='L-BFGS-B', bounds=bounds, options={'maxiter': 5, 'maxfun': 5})
            if res1.success and self.eval_count < self.budget:
                f1 = problem(res1.x)
                self.eval_count += 1
                add_to_archive(res1.x, f1)
                
        if self.eval_count < self.budget - 10:
            # Minimize f2
            def obj2(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                return f[1]
            
            res2 = minimize(obj2, np.random.uniform(lo, hi), method='L-BFGS-B', bounds=bounds, options={'maxiter': 5, 'maxfun': 5})
            if res2.success and self.eval_count < self.budget:
                f2 = problem(res2.x)
                self.eval_count += 1
                add_to_archive(res2.x, f2)

        # Get current bounds of f1 and f2 from archive
        if len(self.archive) == 0:
            return self.archive
            
        f_vals = np.array([f for _, f in self.archive])
        f1_min, f1_max = f_vals[:, 0].min(), f_vals[:, 0].max()
        f2_min, f2_max = f_vals[:, 1].min(), f_vals[:, 1].max()
        
        # Phase 2: Epsilon-Constraint Method
        # We will sweep epsilon for the second objective and minimize the first.
        # Number of steps depends on remaining budget.
        remaining = self.budget - self.eval_count
        n_steps = max(5, remaining // 10) # Allow ~10 evals per step for L-BFGS-B
        
        # Create a grid of epsilon values
        epsilons = np.linspace(f2_min, f2_max, n_steps)
        
        for eps in epsilons:
            if self.eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break
                
            def constrained_obj(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                return f[0]
                
            def constraint(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                # We want f2 <= eps + penalty
                # Using a soft constraint via penalty in objective is often more robust for black-box
                # But L-BFGS-B doesn't handle inequality constraints directly in this simple setup without SLSQP.
                # Let's use SLSQP if available, or just penalty.
                # Since we want to stick to standard scipy, let's try SLSQP for the constrained problem.
                # However, SLSQP is slower. Let's stick to penalty method for speed, or just use L-BFGS-B on penalized objective.
                # Penalized objective: f1 + mu * max(0, f2 - eps)
                # This is non-smooth. Let's use a smooth approximation or just simple penalty.
                # Given the small dimension (4), SLSQP might be okay if we limit iterations.
                pass

            # Let's switch strategy: use SLSQP for the constrained problem
            # Objective: f1
            # Constraint: f2 <= eps
            
            def obj_func(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[0]
                
            def constr_func(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return eps - f[1] # >= 0 means f1 <= eps

            # Initial guess: random or from archive
            x0 = np.random.uniform(lo, hi)
            
            try:
                res = minimize(
                    obj_func, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=[{'type': 'ineq', 'fun': constr_func}],
                    options={'maxiter': 5, 'maxfun': 10}
                )
                if res.success and self.eval_count < self.budget:
                    # Verify the solution
                    f_res = problem(res.x)
                    self.eval_count += 1
                    # Only add if it satisfies the constraint (approximately) and is non-dominated
                    if f_res[1] <= eps + 1e-3:
                         add_to_archive(res.x, f_res)
            except Exception:
                pass

        return self.archive
# EVOLVE-BLOCK-END
