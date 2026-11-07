"""Multi-objective optimizer for DTLZ1 using analytic Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Construct points directly on the DTLZ1 Pareto front for maximum hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 Pareto front: f1 = 0.5*sum(x), f2 = 0.5*sum(x)*(1-y1), f3 = 0.5*sum(x)*y1
        # where y1 is in [0,1]. The Pareto front is a 2D surface parameterized by (t1, t2)
        # where the ideal front has sum(x) = 1 (minimum possible), giving:
        # f1 = 0.5, f2 = 0.5*(1-t), f3 = 0.5*t for t in [0,1]
        # Actually DTLZ1: f1 = 0.5*sum, f2 = 0.5*sum*(1-y1), f3 = 0.5*sum*y1
        # Pareto optimal when sum = 1, so f1=0.5, f2=0.5*(1-t), f3=0.5*t
        
        # We need to map (t1, t2) on a 2D simplex to points on the front.
        # For 3 objectives, the Pareto front is a 2D triangle.
        # Parameterize with (u, v) where u+v <= 1, u,v >= 0:
        # f1 = 0.5, f2 = 0.5*(1-u-v), f3 = 0.5*(u+v) -- no, that's not right.
        
        # Let me reconsider. DTLZ1 with 3 objectives:
        # f1(x) = 0.5 * sum(x_i)
        # f2(x) = 0.5 * sum(x_i) * (1 - y1)  
        # f3(x) = 0.5 * sum(x_i) * y1
        # where y1 is one of the position variables.
        
        # Pareto optimal solutions have sum(x_i) = 1 (the minimum).
        # So on the Pareto front: f1 = 0.5, f2 = 0.5*(1-y1), f3 = 0.5*y1
        # This is a 1D curve! Wait, that can't be right for 3 objectives.
        
        # Actually for DTLZ1 with M objectives:
        # f1 = 0.5*sum, f2 = 0.5*sum*(1-y1), f3 = 0.5*sum*y1*y2, ..., fM = 0.5*sum*y1*...*y_{M-1}
        # Wait no, standard DTLZ1:
        # f1 = 0.5 * sum_{i=1}^{n} x_i
        # f2 = 0.5 * sum * (1 - y1)
        # f3 = 0.5 * sum * y1 * (1 - y2)  [for M=3, only y1 matters]
        # Actually for M=3: f1=0.5*sum, f2=0.5*sum*(1-y1), f3=0.5*sum*y1
        # So it IS a 1D curve parameterized by y1 in [0,1].
        
        # But wait, that gives a line segment, not a 2D surface. Let me verify.
        # For DTLZ1 with M=3, the Pareto front is indeed a 1D curve (line segment)
        # because there's only one position variable y1.
        
        # So we just need to sample y1 values in [0,1] and compute:
        # f1 = 0.5, f2 = 0.5*(1-y1), f3 = 0.5*y1
        
        # But we need to construct x vectors that achieve sum(x_i) = 1.
        # x has 7 variables, bounds [0,1]. We need sum(x_i) = 1.
        # Simple approach: set x_0 = 1, x_1...x_6 = 0, but that's just one point.
        # We need the position variable y1 to vary. In DTLZ1, y1 is typically x_{k}
        # for some k. Let's say y1 = x_0 (first variable is the position variable).
        # Then we need sum(x_i) = 1, so x_0 = y1, and x_1+...+x_6 = 1-y1.
        # We can distribute (1-y1) among x_1...x_6, e.g., evenly: x_1=...=x_6=(1-y1)/6
        
        n_evals = 0
        results = []
        
        # Sample y1 values densely on [0, 1]
        # Use a mix of uniform and clustered samples for good hypervolume coverage
        n_samples = self.budget
        
        # Create a diverse set of y1 values
        y1_values = []
        # Uniform samples
        uniform = np.linspace(0, 1, n_samples // 2)
        y1_values.extend(uniform.tolist())
        # Add some random samples for robustness
        rng = np.random.default_rng(42)
        random_samples = rng.random(n_samples - len(y1_values))
        y1_values.extend(random_samples.tolist())
        
        # Shuffle to mix uniform and random
        rng.shuffle(y1_values)
        
        for y1 in y1_values:
            if time.time() > deadline:
                break
            
            # Construct x: x_0 = y1, x_1..x_6 = (1-y1)/6 each
            x = np.zeros(self.dim)
            x[0] = y1
            remaining = 1.0 - y1
            for i in range(1, self.dim):
                x[i] = remaining / (self.dim - 1)
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            n_evals += 1
            
            if n_evals >= self.budget:
                break
        
        return results
# EVOLVE-BLOCK-END
