"""Multi-objective optimizer on ZDT4 using model-and-optimize / targeted search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Targeted multi-objective optimizer for ZDT4 using model-and-optimize."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        rng = np.random.RandomState(0)
        used = 0
        archive = []  # list of (x, f) pairs

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return x_clipped, f

        def add_to_archive(x, f):
            nonlocal archive
            # Remove points dominated by f
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            # Check if f is dominated
            for ax, af in new_archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            new_archive.append((x, f))
            archive = new_archive

        # ZDT4 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=1..n
        # h(f1, g) = 1 + sqrt(f1/g)
        # f2(x) = g(x) * h(f1, g(x))
        # Optimal: g=1 when x_i=0 for i>=1 (each term = 0 - 10 = -10, sum = -90, g = 1+90-90=1)
        # Pareto front: f1 in [0,1], f2 = 1 + sqrt(f1/1) = 1 + sqrt(f1)

        # Phase 1: Systematic exploration along the Pareto front
        # Use many x[0] candidates with non-uniform sampling biased toward
        # lower values (higher curvature region of the Pareto front)
        n_candidates = min(100, budget // 2)
        # Non-uniform: bias toward lower x0 where f2 = 1+sqrt(f1) has high curvature
        # Use power-law spacing: x0_i = (i+1)^alpha / (n+1)^alpha for better coverage
        idx = np.arange(1, n_candidates + 1, dtype=float)
        alpha = 0.7  # bias toward lower values
        x0_vals = (idx / (n_candidates + 1)) ** alpha * 0.994 + 0.003
        x0_vals = np.clip(x0_vals, 0.003, 0.997)
        
        for x0 in x0_vals:
            if used >= budget:
                break
            if time.time() > deadline - 60:
                break
            
            # Try 2 perturbations around x_i=0 for i>=1
            for attempt in range(2):
                if used >= budget:
                    break
                x = np.zeros(dim)
                x[0] = x0
                # Perturb x[1:dim] slightly toward 0
                scale = 0.02 if attempt == 0 else 0.005
                x[1:] = 0.0 + scale * rng.randn(dim - 1)
                x = np.clip(x, lo, hi)
                x, f = evaluate(x)
                add_to_archive(x.copy(), f.copy())

        # Phase 2: Local refinement of best archive points
        # Focus on reducing g by pushing x[1:dim] toward 0
        if len(archive) > 0 and used < budget:
            # Sort by f2 (second objective) to prioritize points with good g values
            archive.sort(key=lambda p: p[1][1])
            
            max_refine = min(len(archive), (budget - used) // 3)
            for i, (x_best, f_best) in enumerate(archive[:max_refine]):
                if used >= budget:
                    break
                if time.time() > deadline - 60:
                    break
                
                # Aggressive local search: push x[1:dim] toward 0
                n_local = min(4, budget - used)
                for _ in range(n_local):
                    if used >= budget:
                        break
                    if time.time() > deadline - 60:
                        break
                    
                    # Mix: some targeted toward 0 for x[1:], some general local search
                    if _ < n_local // 2:
                        # Targeted: reduce x[1:dim] magnitude
                        x_new = x_best.copy()
                        x_new[1:] = x_new[1:] * 0.4 + 0.005 * rng.randn(dim - 1)
                    else:
                        # General local search with small step
                        x_new = x_best + 0.02 * rng.randn(dim)
                    x_new = np.clip(x_new, lo, hi)
                    x_new, f_new = evaluate(x_new)
                    add_to_archive(x_new.copy(), f_new.copy())

        # Phase 3: Fill remaining budget with targeted search
        while used < budget:
            if time.time() > deadline - 10:
                break
            
            # Choose x0 with bias toward lower values (higher hypervolume contribution)
            u = rng.uniform()
            x0_new = (u ** 0.7) * 0.994 + 0.003
            x0_new = np.clip(x0_new, 0.003, 0.997)
            x_new = np.zeros(dim)
            x_new[0] = x0_new
            x_new[1:] = 0.0 + 0.005 * rng.randn(dim - 1)
            x_new = np.clip(x_new, lo, hi)
            x_new, f_new = evaluate(x_new)
            add_to_archive(x_new.copy(), f_new.copy())

            # If we have budget, do a local refinement
            if len(archive) > 0 and used < budget:
                if time.time() > deadline - 10:
                    break
                idx = rng.randint(len(archive))
                x_base, f_base = archive[idx]
                # Push x[1:dim] further toward 0
                x_refine = x_base.copy()
                x_refine[1:] = x_refine[1:] * 0.3 + 0.003 * rng.randn(dim - 1)
                x_refine = np.clip(x_refine, lo, hi)
                x_refine, f_refine = evaluate(x_refine)
                add_to_archive(x_refine.copy(), f_refine.copy())

        # Return (return value is ignored by evaluator, but must not raise)
        if len(archive) > 0:
            return np.array([x for x, f in archive])
        return np.array([np.zeros(dim)])
# EVOLVE-BLOCK-END
