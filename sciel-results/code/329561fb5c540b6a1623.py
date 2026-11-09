"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search for WFG4 with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = 50  # Fixed reasonable size for 200 budget

        # Initialize population using Latin Hypercube Sampling for better coverage
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(pop_size)
            offsets = rng.random(pop_size)
            X[:, d] = (perm + offsets) / pop_size
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        def is_non_dominated_single(f, F):
            """Check if f is non-dominated w.r.t F."""
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return False
            return True

        def dominated_indices(f, F):
            """Return indices of points in F dominated by f."""
            dom_idx = []
            for i, fi in enumerate(F):
                if np.all(f <= fi) and np.any(f < fi):
                    dom_idx.append(i)
            return dom_idx

        # Main evolutionary phase
        evolve_budget = int(self.budget * 0.8)
        refine_budget = self.budget - evolve_budget
        
        F_factor = 0.8  # Differential Evolution scale factor

        while used < evolve_budget:
            if time.time() > deadline:
                break

            # Differential Evolution Mutation
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            v = X[r1] + F_factor * (X[r2] - X[r3])
            v = np.clip(v, lo, hi)

            fv = problem(v)
            used += 1

            # Selection
            if is_non_dominated_single(fv, F):
                # fv is non-dominated
                dom_idx = dominated_indices(fv, F)
                if dom_idx:
                    # Replace a dominated point
                    # Pick the one with the worst sum among dominated
                    sums = np.sum(F[dom_idx], axis=1)
                    worst_dom_local = np.argmax(sums)
                    replace_idx = dom_idx[worst_dom_local]
                else:
                    # No dominated points. Replace the point with the worst sum overall
                    sums = np.sum(F, axis=1)
                    replace_idx = np.argmax(sums)
                
                X[replace_idx] = v
                F[replace_idx] = fv

        # Local refinement phase: polish non-dominated points with small perturbations
        # Find non-dominated points in current population
        non_dom_mask = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break
        
        nd_indices = np.where(non_dom_mask)[0]
        
        if len(nd_indices) == 0:
            nd_indices = np.arange(pop_size)

        # Sort non-dominated points by sum (best first)
        nd_sums = np.sum(F[nd_indices], axis=1)
        nd_sorted = nd_indices[np.argsort(nd_sums)]

        # Refine top points
        refine_scale = 0.02
        points_to_refine = nd_sorted[:min(10, len(nd_sorted))]

        for pi in points_to_refine:
            if used >= self.budget or time.time() > deadline:
                break

            # Try several small perturbations around this point
            for _ in range(5): # Increase attempts per point
                if used >= self.budget or time.time() > deadline:
                    break

                child = X[pi] + refine_scale * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # If this improves (is non-dominated), add to population
                if is_non_dominated_single(fc, F):
                    dom_idx = dominated_indices(fc, F)
                    if dom_idx:
                        sums = np.sum(F[dom_idx], axis=1)
                        worst_dom_local = np.argmax(sums)
                        replace_idx = dom_idx[worst_dom_local]
                    else:
                        sums = np.sum(F, axis=1)
                        replace_idx = np.argmax(sums)
                    
                    X[replace_idx] = child
                    F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
