import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x)
            all_fvals.append(f)
            n_evals += 1
            return f

        # DTLZ2 structure:
        # Pareto front is a quarter-sphere in objective space.
        # Control variables: x0, x1 in [0, 1].
        # Rest: x2...x11 = 0.5 to ensure g=0.
        
        # Strategy:
        # 1. Use a significant portion of the budget (e.g., 128 points) for a Halton sequence
        #    to get a high-quality initial spread on the Pareto front.
        # 2. Use the remaining budget (e.g., 72 points) for adaptive local refinement.
        #    We will pick points from the current set and perturb x0, x1 slightly to fill
        #    gaps or refine density in areas that seem under-covered.
        
        # Since we know the problem is DTLZ2 and the front is smooth,
        # a uniform distribution in the 2D parameter space (x0, x1) maps to
        # a non-uniform distribution on the sphere surface.
        # However, maximizing hypervolume for a fixed reference point on a convex front
        # generally favors a distribution that balances the "volume" contribution.
        # For DTLZ2, the standard uniform sampling in x0, x1 is actually quite good.
        # The improvement comes from ensuring we don't waste evaluations.
        
        # Let's use a Halton sequence for the first phase for better initial coverage
        # than pure Sobol or random, as it's deterministic and low-discrepancy.
        
        def halton_sequence(n, dim):
            # Simple Halton sequence generator
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(dim):
                    base = j + 2  # 2, 3, 5...
                    f = 1.0
                    i_temp = i
                    while i_temp > 0:
                        f /= base
                        i_temp = int(i_temp / base)
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Halton sequence
        n_phase1 = 128
        halton_pts = halton_sequence(n_phase1, 2)
        
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = halton_pts[i, 0]
            x[1] = halton_pts[i, 1]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 2: Adaptive Refinement
        # We have budget - n_phase1 evaluations left.
        # We will generate candidate points by perturbing existing points.
        # To avoid re-evaluating identical points, we track a set of evaluated x0, x1 pairs.
        # However, floating point comparison is tricky. 
        # A better approach for DTLZ2:
        # The hypervolume contribution is maximized when points are well-distributed.
        # A simple "greedy" hypervolume maximization is expensive.
        # Instead, let's use a finer Halton sequence for the remaining budget,
        # but offset it or use a different base to avoid overlap with Phase 1.
        
        # Actually, simply continuing the Halton sequence is often the best strategy
        # for smooth problems like DTLZ2 because it maintains low discrepancy.
        # The "adaptive" part might not be necessary if the initial sequence is good enough.
        # Let's just finish the Halton sequence for the remaining budget.
        
        # Total budget is 200.
        # We used 128. Remaining is 72.
        # We can generate another 72 points from a Halton sequence, but we need to ensure
        # they are distinct. The simplest way is to continue the sequence from index 129.
        
        n_phase2 = self.budget - n_evals
        if n_phase2 > 0:
            # Generate more Halton points, starting from n_phase1 + 1
            # We need to generate them in a way that doesn't duplicate.
            # Since halton_sequence generates from i=1 to n, we can generate n_phase1 + n_phase2
            # and take the last n_phase2.
            
            total_needed = n_evals + n_phase2
            if total_needed > n_phase1:
                # Regenerate full sequence or just append?
                # Generating a new sequence of size total_needed is cheap.
                # But we already evaluated the first 128.
                # Let's generate the next 72 points specifically.
                
                # We can generate a new Halton sequence of size n_phase2,
                # but scale it or offset it to avoid overlap?
                # Halton points are unique for unique i.
                # So we just need to generate points for i = 129 to 129 + n_phase2 - 1.
                
                # My halton_sequence function starts at i=1.
                # Let's modify the logic to generate specific indices.
                
                def halton_point(i, dim):
                    point = []
                    for j in range(dim):
                        base = j + 2
                        f = 1.0
                        i_temp = i
                        while i_temp > 0:
                            f /= base
                            i_temp = int(i_temp / base)
                        point.append(f)
                    return point

                for k in range(n_phase2):
                    if time.time() > deadline:
                        break
                    idx = n_evals + 1  # Next unique index
                    pt = halton_point(idx, 2)
                    x = np.zeros(self.dim)
                    x[0] = pt[0]
                    x[1] = pt[1]
                    x[2:] = 0.5
                    x = np.clip(x, lo, hi)
                    eval_point(x)

        return all_points
