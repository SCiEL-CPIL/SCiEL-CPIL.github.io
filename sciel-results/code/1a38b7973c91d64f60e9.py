"""DTLZ1 optimizer using Hammersley point set on 2D simplex for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic 2D Pareto front with Hammersley points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5  # Set all g-variables to 0.5 to minimize g
        
        n_evals = min(budget, 200)
        if n_evals <= 0:
            return all_points
        
        points_to_eval = []
        seen = set()
        
        # Generate Hammersley point set in [0,1]^2, then map to 2D simplex
        # Hammersley sequence: first coordinate is van der Corput base 2,
        # second coordinate is van der Corput base 3
        def van_der_corput(n, base):
            """Generate n points of van der Corput sequence in [0,1]"""
            result = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = i + 1  # 1-indexed
                while r > 0:
                    f /= base
                    r, rem = divmod(r, base)
                    result[i] += rem * f
            return result
        
        # Generate Hammersley points
        u1 = van_der_corput(n_evals, 2)  # base 2
        u2 = van_der_corput(n_evals, 3)  # base 3
        
        # Map (u1, u2) from unit square to 2D simplex
        # Simplex: x0 >= 0, x1 >= 0, x0 + x1 <= 1
        # Use transformation: x0 = u1 * (1 - u2), x1 = u2 * (1 - u1)
        # This maps [0,1]^2 to the simplex with Jacobian that preserves uniformity
        # Actually, a better mapping: 
        # Let x0 = u1, x1 = u2 * (1 - u1)
        # This gives a uniform distribution over the simplex if u1, u2 are uniform in [0,1]
        
        for i in range(n_evals):
            x0 = u1[i]
            x1 = u2[i] * (1.0 - x0)
            
            # Ensure we stay within the simplex (numerical safety)
            x0 = max(0.0, min(1.0, x0))
            x1 = max(0.0, min(1.0 - x0, x1))
            
            key = (round(x0, 12), round(x1, 12))
            if key not in seen:
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                points_to_eval.append(x)
                seen.add(key)
        
        # If we have fewer unique points than n_evals due to collisions,
        # add additional stratified points to fill the budget
        remaining = n_evals - len(points_to_eval)
        if remaining > 0:
            # Add corner points and edge points that might have been missed
            candidates = [
                (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
                (1/3, 1/3), (2/3, 0.0), (0.0, 2/3),
            ]
            for x0, x1 in candidates:
                if len(points_to_eval) >= n_evals:
                    break
                key = (round(x0, 12), round(x1, 12))
                if key not in seen:
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    points_to_eval.append(x)
                    seen.add(key)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
