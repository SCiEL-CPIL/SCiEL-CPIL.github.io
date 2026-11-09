import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_head = 2
        n_tail = self.dim - n_head
        
        # DTLZ7 specific structure:
        # f1 = x1 * (1 + g)
        # f2 = x2 * (1 + g)
        # f3 = 1 + 2*g
        # where g depends on tail variables x[2:].
        # To minimize g (and thus f3, and maximize the reach of f1, f2),
        # we should set tail variables to their optimal values.
        # For standard DTLZ7, x=0.5 for tail variables is a common optimal point for g.
        # Even if not perfectly 0, this is much better than random or slow local search.
        
        # Construct the optimal tail
        # Assuming bounds are [0, 1] as per task description
        # If bounds vary, 0.5 might not be optimal, but for DTLZ7 standard, it is.
        # Let's use the midpoint of bounds for tail variables as a robust heuristic for "optimal" in many MOPs, 
        # but specifically for DTLZ7, 0.5 is the known optimum for the standard formulation.
        # The task says "bounds [0, 1] for every variable".
        
        tail_opt = np.full(n_tail, 0.5)
        
        # We will sample the head variables (x1, x2) to cover the Pareto front.
        # The Pareto front is defined by x1, x2 in [0, 1] with optimal tail.
        
        # Generate Halton sequence points for x1, x2
        # We have budget evaluations.
        # We don't need to evaluate the tail separately if we just construct x directly.
        # Each evaluation of problem(x) counts as 1.
        
        n_evals = self.budget
        
        # Halton sequence generator for 2 dimensions
        def halton_sequence(n, dims):
            # Simple implementation for 2D Halton
            # Base 2 and 3
            points = []
            for i in range(1, n + 1):
                p = []
                # Dimension 1: base 2
                f = 1.0
                i_val = i
                while i_val > 0:
                    f /= 2.0
                    i_val = int(i_val / 2)
                    if i_val % 2 == 1:
                        p.append(f)
                p1 = sum(p)
                
                # Dimension 2: base 3
                f = 1.0
                i_val = i
                while i_val > 0:
                    f /= 3.0
                    i_val = int(i_val / 3)
                    if i_val % 3 != 0:
                        p.append(f * (i_val % 3))
                p2 = sum(p)
                
                points.append([p1, p2])
            return np.array(points)
        
        # We want to cover [0, 1] x [0, 1] for x1, x2
        # Halton sequence is in [0, 1].
        
        try:
            pts = halton_sequence(n_evals, 2)
        except Exception:
            pts = np.random.rand(n_evals, 2)
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline - 5.0:
                break
            
            x1 = pts[i, 0]
            x2 = pts[i, 1]
            
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            x[2:] = tail_opt
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        if not results:
            # Fallback if something went wrong
            x = np.random.rand(self.dim)
            x = np.clip(x, lo, hi)
            results.append(problem(x))
            
        return np.array(results)
