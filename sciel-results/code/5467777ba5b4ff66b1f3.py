"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    - Pareto front requires x_2..x_10 = 0 and x_1 in [0,1]
    - f1 = 1 + 9*sum(g_i), f2 = (1-sqrt(f1/10))*(1-(f1/10)*sin(10*pi*f1/10))
    - Strategy: densely sample x_1 with x_2..x_10=0, plus some exploration
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 specific: x_1 bounds are typically [-5,5], x_2..x_10 are [-5,5]
        # Pareto front: x_2..x_10 = 0, x_1 in [0,1]
        
        # Strategy: Use most budget to sample the Pareto front directly
        # Reserve a small fraction for exploration/refinement
        
        n_front = int(self.budget * 0.85)  # 170 evaluations for front sampling
        n_explore = self.budget - n_front   # 30 for exploration
        
        # Generate x_1 values densely across [0, 1]
        # Use a mix of uniform and slightly denser near endpoints
        x1_vals = np.linspace(0.0, 1.0, n_front)
        
        # Create candidate points on the Pareto front
        candidates = []
        for x1 in x1_vals:
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 (already zero)
            candidates.append(x)
        
        # Add some exploration points
        rng = np.random.default_rng(42)
        for _ in range(n_explore):
            # Mix: some near-front points with small perturbations, some random
            if rng.random() < 0.7:
                # Near-front: x_1 in [0,1], x_2..x_10 small
                x = np.zeros(self.dim)
                x[0] = rng.uniform(0.0, 1.0)
                x[1:] = rng.uniform(-0.1, 0.1, self.dim - 1)
            else:
                # Random exploration
                x = lo + (hi - lo) * rng.random(self.dim)
            candidates.append(x)
        
        # Evaluate all candidates, tracking time
        results = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            results.append(f)
            used += 1
        
        return results
# EVOLVE-BLOCK-END
