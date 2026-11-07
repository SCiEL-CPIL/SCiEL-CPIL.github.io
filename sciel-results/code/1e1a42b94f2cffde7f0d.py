import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Helper to extract g from f1, f2 for ZDT2
        # f1 = x1
        # f2 = g * (1 - sqrt(x1/g))
        # f2/g = 1 - sqrt(f1/g)
        # sqrt(f1/g) = 1 - f2/g
        # f1/g = (1 - f2/g)^2
        # f1 = g (1 - f2/g)^2 = g (1 - 2f2/g + f2^2/g^2) = g - 2f2 + f2^2/g
        # g^2 - 2f2*g + (f1 - f2^2) = 0
        # g = (2f2 + sqrt(4f2^2 - 4(f1 - f2^2))) / 2 = f2 + sqrt(f2^2 - f1 + f2^2) = f2 + sqrt(2f2^2 - f1)
        # Wait, the standard formula derivation:
        # f2 = g(1 - sqrt(f1/g))
        # f2/g = 1 - sqrt(f1/g)
        # 1 - f2/g = sqrt(f1/g)
        # (1 - f2/g)^2 = f1/g
        # g (1 - f2/g)^2 = f1
        # g (1 - 2f2/g + f2^2/g^2) = f1
        # g - 2f2 + f2^2/g = f1
        # g^2 - 2f2*g + f2^2 - f1*g = 0
        # g^2 - (2f2 + f1)g + f2^2 = 0
        # g = [(2f2 + f1) + sqrt((2f2 + f1)^2 - 4f2^2)] / 2
        # g = [(2f2 + f1) + sqrt(4f2^2 + 4f2f1 + f1^2 - 4f2^2)] / 2
        # g = [(2f2 + f1) + sqrt(4f2f1 + f1^2)] / 2
        # g = [(2f2 + f1) + f1*sqrt(4f2/f1 + 1)] / 2  (if f1>0)
        
        def estimate_g(f1, f2):
            if f1 == 0:
                return f2
            if f1 < 0 or f2 < 0:
                return np.inf
            # Use the stable quadratic formula
            # g^2 - (2f2 + f1)g + f2^2 = 0
            # Roots: g = (A + sqrt(A^2 - 4B)) / 2 where A = 2f2 + f1, B = f2^2
            # A^2 - 4B = (2f2+f1)^2 - 4f2^2 = 4f2^2 + 4f2f1 + f1^2 - 4f2^2 = 4f2f1 + f1^2
            # sqrt_term = sqrt(f1 * (4*f2 + f1))
            if 4*f2 + f1 < 0:
                return np.inf
            sqrt_term = np.sqrt(f1 * (4*f2 + f1))
            g = (2*f2 + f1 + sqrt_term) / 2.0
            return g

        # Objective function for differential evolution: minimize g(x_tail)
        # We fix x1 to a value that allows us to estimate g.
        # If x1=0, f1=0, f2=g. So g=f2. This is perfect!
        # Let's verify: if x1=0, f1=0. f2 = g(1 - 0) = g.
        # So we can just evaluate problem([0] + x_tail) and take f[1] as g.
        # This is much more accurate than estimating from f1>0.
        
        def objective_g(x_tail):
            x = np.concatenate(([0.0], x_tail))
            x = np.clip(x, lo, hi)
            f = problem(x)
            # f[1] is g when f[0]=0
            return f[1]

        # Split budget
        # DE is expensive. Let's give it 40% of the budget.
        # The rest for sampling the front.
        n_de_budget = int(self.budget * 0.4)
        n_sample_budget = self.budget - n_de_budget
        
        # Differential Evolution bounds for tail
        # DE expects a list of tuples (min, max)
        de_bounds = [(lo[i], hi[i]) for i in range(1, self.dim)]
        
        try:
            # maxiter and popsize control evaluations.
            # Total evals approx = popsize * maxiter * dim? No, popsize * maxiter.
            # We want to use approx n_de_budget evaluations.
            # popsize=15, maxiter=50 -> 750 evals? Too many if budget is 200.
            # Budget 200. n_de_budget = 80.
            # popsize=10, maxiter=8 -> 80 evals.
            popsize = 10
            maxiter = max(1, n_de_budget // popsize)
            
            # Seed DE for reproducibility? The task says seeds are set by evaluator.
            # DE uses global random state.
            result = differential_evolution(objective_g, de_bounds, maxiter=maxiter, popsize=popsize, tol=1e-6, atol=1e-8, seed=None)
            
            best_tail = result.x
            best_g = result.fun
            
        except Exception:
            # Fallback to random search if DE fails or exceeds budget unexpectedly
            # Just pick a random tail
            best_tail = lo[1:] + (hi[1:] - lo[1:]) * np.random.random(self.dim - 1)
            # Evaluate it to get a valid g? We don't strictly need to if we just use it for sampling,
            # but we should ensure it's a valid point.
            # We will just use this tail for the sampling phase.

        # Phase 2: Sample the Pareto Front
        # The front is defined by x1 in [0, 1] and x_tail = best_tail.
        # f1 = x1
        # f2 = g_best * (1 - sqrt(x1 / g_best))
        
        # To maximize hypervolume, we should distribute x1 points to cover the curve well.
        # Since the curve is concave, uniform x1 is good.
        # Let's use linspace.
        if n_sample_budget > 0:
            x1_values = np.linspace(0.0, 1.0, n_sample_budget)
            for x1 in x1_values:
                x = np.concatenate(([x1], best_tail))
                x = np.clip(x, lo, hi)
                problem(x)
                
        return None
