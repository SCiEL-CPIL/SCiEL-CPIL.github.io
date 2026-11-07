"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # DTLZ2 structure: x[0] is the decision variable for the front, x[1:] are position variables
        # f_i = (1 + g(x[1:])) * (1 - sin(pi * x[0] / 2))
        # g(x[1:]) = 100 * (n-1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))))
        # To minimize, we need g=0 and sin(pi*x[0]/2) = 1 => x[0] = 1.0
        # For g=0, we need x_i = 0.5 for all i in x[1:]
        
        # However, the problem might be scaled or shifted. Let's use a hybrid approach.
        # 1. Try the known optimum first.
        # 2. If that fails or if we have budget, do a local search around the known optimum.
        # 3. Also do a random search to handle any unexpected scaling.
        
        # Known optimum for standard DTLZ2: x[0]=1, x[1:]=0.5
        # But bounds are [0,1].
        
        # Let's evaluate a set of promising points first.
        # Point 1: Theoretical optimum
        x_opt = np.full(dim, 0.5)
        x_opt[0] = 1.0
        f_opt = problem(x_opt)
        
        # Check if this is valid (all objectives < reference point 2.5)
        # If f_opt is good, we can do a local search.
        
        # Strategy:
        # 1. Evaluate the theoretical optimum.
        # 2. Use remaining budget for a local search around the optimum using Nelder-Mead or similar,
        #    but since it's multi-objective, we can just do a random walk with small steps
        #    and keep the best non-dominated points.
        
        # Actually, for DTLZ2, the Pareto front is a hypersphere. 
        # The objective values are f_i = (1+g) * sin(pi * x_i / 2) for i=0..m-1? No.
        # DTLZ2: m objectives.
        # f_1 = (1+g) * sin(pi * x_1 / 2)
        # f_2 = (1+g) * cos(pi * x_1 / 2) * sin(pi * x_2 / 2)
        # ...
        # f_m = (1+g) * cos(pi * x_{m-1} / 2)
        
        # Wait, the standard DTLZ2 has m objectives. Here n_obj=3.
        # So x[0] and x[1] are the front variables. x[2:] are position variables.
        # f_1 = (1+g) * sin(pi * x_0 / 2)
        # f_2 = (1+g) * cos(pi * x_0 / 2) * sin(pi * x_1 / 2)
        # f_3 = (1+g) * cos(pi * x_0 / 2) * cos(pi * x_1 / 2)
        
        # To minimize, we want g=0.
        # g = 100 * (n-1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))) for i in x[2:]
        # Minimum g=0 when x_i = 0.5 for all i in x[2:].
        
        # So the Pareto front is defined by x_0, x_1 in [0,1] and x[2:]=0.5.
        # f_1 = sin(pi * x_0 / 2)
        # f_2 = cos(pi * x_0 / 2) * sin(pi * x_1 / 2)
        # f_3 = cos(pi * x_0 / 2) * cos(pi * x_1 / 2)
        
        # This is a quarter sphere of radius 1 in the first octant.
        # We need to cover this front well.
        
        # Let's generate points on this front.
        # We can use a grid or random sampling on the sphere.
        
        # First, evaluate the theoretical optimum for g=0.
        # x[0]=1, x[1]=0.5, x[2:]=0.5
        # f = [sin(pi/2), cos(pi/2)*sin(pi/4), cos(pi/2)*cos(pi/4)] = [1, 0, 0]
        
        # x[0]=0, x[1]=1, x[2:]=0.5
        # f = [0, 1, 0]
        
        # x[0]=0, x[1]=0, x[2:]=0.5
        # f = [0, 0, 1]
        
        # So the front is the set of points (sin(pi*x0/2), cos(pi*x0/2)*sin(pi*x1/2), cos(pi*x0/2)*cos(pi*x1/2))
        
        # Let's sample this front.
        # We have budget evaluations.
        
        # 1. Sample the front with a grid.
        # 2. Use remaining budget for local search to ensure g=0.
        
        # Let's try to evaluate a grid on the front.
        # Grid size: sqrt(budget/2) ?
        
        # Actually, let's just do a smart random search.
        # 1. Generate random points on the front (x[2:]=0.5).
        # 2. Check if g is close to 0. If not, do a local search to reduce g.
        
        # But we don't know if the problem is exactly DTLZ2 or a variant.
        # The task says "minimizing all 3 objectives of DTLZ2".
        
        # Let's assume it is standard DTLZ2.
        
        # Plan:
        # 1. Evaluate a set of points on the theoretical front (x[2:]=0.5).
        # 2. If any point has objectives > 1.0 (indicating g > 0), do a local search to minimize g.
        
        # Let's start by evaluating the theoretical front points.
        # We can use a Sobol sequence or random uniform for x0, x1.
        
        rng = np.random.default_rng(0)
        
        # Number of points to evaluate on the front
        n_front = min(budget // 2, 100)
        
        # Generate x0, x1 in [0, 1]
        x0 = rng.random(n_front)
        x1 = rng.random(n_front)
        
        X = np.zeros((n_front, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5
        
        F = np.array([problem(x) for x in X])
        used = n_front
        
        # Check if we need to reduce g.
        # If any f_i > 1.0, then g > 0.
        # Let's find the point with the smallest sum of objectives.
        # If min sum > 1.0, we need to reduce g.
        
        # Actually, let's just do a local search on the best point.
        # But local search for multi-objective is tricky.
        # We can just minimize the sum of objectives.
        
        # Let's use a simple gradient-free local search.
        # Start from the best point found so far.
        
        best_idx = np.argmin(np.sum(F, axis=1))
        x_best = X[best_idx].copy()
        f_best = F[best_idx].copy()
        
        # Local search
        step_size = 0.1
        n_local = budget - used - 10  # Leave some budget for safety
        
        for i in range(n_local):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            x_new = x_best + step_size * rng.standard_normal(dim)
            x_new = np.clip(x_new, lo, hi)
            
            # Ensure x[2:] are close to 0.5 to keep g small
            # Actually, let's just evaluate it.
            f_new = problem(x_new)
            used += 1
            
            # If f_new is better than f_best (in terms of sum), accept it.
            if np.sum(f_new) < np.sum(f_best):
                x_best = x_new
                f_best = f_new
                step_size *= 0.9  # Reduce step size
            else:
                step_size *= 0.95  # Reduce step size slightly
        
        # Return the best point found.
        # But the task says "Every returned objective vector is recorded".
        # So we should return a list of points, or the last point?
        # "The return value of __call__ is ignored."
        # So we can return anything.
        
        return x_best
# EVOLVE-BLOCK-END
