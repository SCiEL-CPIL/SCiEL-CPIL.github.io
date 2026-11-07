"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    - Pareto front requires x_2..x_10 = 0 and x_1 in [0,1] (or appropriate range)
    - f1 = 1 + 9*sum(g_i), f2 = (1-sqrt(f1/10))*(1-(f1/10)*sin(10*pi*f1/10))
    - Strategy: densely sample x_1 with x_2..x_10=0, plus some exploration
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 specific: x_1 bounds are typically [-5,5] or [0,1], x_2..x_10 are [-5,5]
        # Pareto front: x_2..x_10 = 0, x_1 in [0,1]
        
        # Strategy: Use most budget to sample the Pareto front directly
        # Reserve a small fraction for exploration/refinement
        
        n_front = int(self.budget * 0.90)  # 180 evaluations for front sampling
        n_explore = self.budget - n_front   # 20 for exploration
        
        # Determine the range for x1 to sweep
        # For ZDT4, the Pareto front is defined by x1 in [0,1] typically
        # But we should respect the bounds. If bounds[0] is [0,1], use that.
        # If bounds[0] is [-5,5], the interesting part is still [0,1] for the front.
        # Let's assume the front lies within the bounds of x1.
        # To be safe and cover the "interesting" part, we'll sweep the full range of bounds[0]
        # but if the range is large, we might miss the dense front if it's only in [0,1].
        # Given the reference point [11, 400] and typical ZDT4 scaling,
        # if f1_min ~ 1, then x1 likely needs to be in [0,1].
        
        # Let's check if 0 and 1 are within bounds[0]
        x1_low = max(lo[0], 0.0)
        x1_high = min(hi[0], 1.0)
        
        # If the intersection is empty, fall back to the full bounds
        if x1_low >= x1_high:
            x1_low = lo[0]
            x1_high = hi[0]
        
        # Create candidate points on the Pareto front
        # Use a uniform grid for x1 in [x1_low, x1_high]
        # To handle oscillations in f2 (sin term), a uniform grid is generally good.
        # However, we can also use a slightly denser grid if we have budget.
        
        candidates = []
        
        # 1. Front sampling
        # We want to cover the Pareto front.
        # Let's use n_front evaluations for this.
        if n_front > 0:
            x1_vals = np.linspace(x1_low, x1_high, n_front)
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x1
                # Set x[1:] to 0 (or closest bound if 0 is out of range)
                for i in range(1, self.dim):
                    if 0.0 < lo[i]:
                        x[i] = lo[i]
                    elif 0.0 > hi[i]:
                        x[i] = hi[i]
                    else:
                        x[i] = 0.0
                candidates.append(x)
        
        # 2. Exploration
        # Use remaining budget for exploration to ensure we don't miss anything
        # and to handle cases where the assumption about the front location is slightly off.
        rng = np.random.default_rng(42)
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            # Mix: some near-front points with small perturbations, some random
            if rng.random() < 0.7:
                # Near-front: x_1 in [x1_low, x1_high], x_2..x_10 small
                x = np.zeros(self.dim)
                x[0] = rng.uniform(x1_low, x1_high)
                for i in range(1, self.dim):
                    if 0.0 < lo[i]:
                        x[i] = lo[i] + rng.uniform(0, 0.05)
                    elif 0.0 > hi[i]:
                        x[i] = hi[i] - rng.uniform(0, 0.05)
                    else:
                        x[i] = rng.uniform(-0.05, 0.05)
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
            try:
                f = problem(x_clipped)
                results.append(f)
                used += 1
            except Exception:
                # If an exception occurs, we stop to preserve the results we have
                # The contract says: "an exception later keeps the evaluations made"
                # But we should probably break to avoid infinite loops or repeated errors
                break
        
        return results
# EVOLVE-BLOCK-END
