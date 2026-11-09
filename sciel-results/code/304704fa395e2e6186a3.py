"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct-structure optimizer for WFG7 with hypervolume-aware search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s/330s limits

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        n_obj = self.n_obj
        dim = self.dim

        # Helper functions
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_set(F):
            idx = []
            for i, f in enumerate(F):
                dominated = False
                for j, g in enumerate(F):
                    if i != j and dominates(g, f):
                        dominated = True
                        break
                if not dominated:
                    idx.append(i)
            return np.array(idx, dtype=int)

        # Phase 1: Dense random sampling to get initial front
        n_init = min(budget, 60)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Reference point for dominance/hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Phase 2: Iterative local refinement using simplex-like steps
        pop_size = 12
        nd_idx = non_dominated_set(F)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])

        X_ref = X[nd_idx[:pop_size]].copy()
        F_ref = F[nd_idx[:pop_size]].copy()
        if len(X_ref) < pop_size:
            n_pad = pop_size - len(X_ref)
            X_pad = lo + (hi - lo) * rng.random((n_pad, dim))
            F_pad = np.array([problem(x) for x in X_pad])
            used += n_pad
            X_ref = np.vstack([X_ref, X_pad])
            F_ref = np.vstack([F_ref, F_pad])

        # Adaptive step size
        step_scale = 0.15 * (hi - lo)

        # Phase 3: Main optimization loop with local search
        max_iter = budget - used
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Select parent: random non-dominated from refined pop
            nd_in_pop = []
            for i in range(len(F_ref)):
                if not any(dominates(F_ref[j], F_ref[i]) for j in range(len(F_ref)) if j != i):
                    nd_in_pop.append(i)
            if not nd_in_pop:
                nd_in_pop = list(range(len(F_ref)))

            parent_idx = nd_in_pop[rng.integers(len(nd_in_pop))]
            parent = X_ref[parent_idx].copy()

            # Mutation: random direction with adaptive step
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-10
            # Mix of small and larger steps
            scale = step_scale * (1.0 + 0.5 * rng.random())
            child = parent + scale * direction
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update refined population
            dominated_mask = np.array([any(dominates(F_ref[j], F_ref[i]) for j in range(len(F_ref)) if j != i) for i in range(len(F_ref))])
            if np.any(dominated_mask):
                worst_idx = np.where(dominated_mask)[0][rng.integers(np.sum(dominated_mask))]
                X_ref[worst_idx] = child
                F_ref[worst_idx] = fc
            else:
                F_norm = (F_ref - np.min(F_ref, axis=0)) / (np.max(F_ref, axis=0) - np.min(F_ref, axis=0) + 1e-10)
                worst_idx = int(np.argmax(np.sum(F_norm, axis=1)))
                if np.any(fc < F_ref[worst_idx]):
                    X_ref[worst_idx] = child
                    F_ref[worst_idx] = fc

            # Periodically add new random exploration points
            if it % 20 == 0 and used + 1 <= budget:
                exp = lo + (hi - lo) * rng.random(dim)
                fe = problem(exp)
                used += 1
                if not any(dominates(F_ref[j], fe) for j in range(len(F_ref))):
                    ridx = rng.integers(len(F_ref))
                    X_ref[ridx] = exp
                    F_ref[ridx] = fe

        # Phase 4: Local refinement around non-dominated points
        # Use remaining budget for local search on the best points
        all_F = np.vstack([F, F_ref])
        all_X = np.vstack([X, X_ref])
        
        # Get non-dominated indices from all evaluated points
        final_nd = non_dominated_set(all_F)
        
        # For each non-dominated point, do a local refinement
        remaining = budget - used
        if remaining > 0 and len(final_nd) > 0:
            # Sort non-dominated points by their distance to reference point (closer is better)
            nd_points = all_F[final_nd]
            dists = np.sum((nd_points - ref) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            
            # Allocate remaining budget to refine the best non-dominated points
            n_refine = min(len(final_nd), max(1, remaining // 10))
            
            for i in range(n_refine):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Get the point to refine
                orig_idx = final_nd[sorted_idx[i]]
                base_x = all_X[orig_idx].copy()
                base_f = all_F[orig_idx].copy()
                
                # Local search: try small perturbations
                local_budget = min(10, budget - used)
                local_step = 0.02 * (hi - lo)
                
                for j in range(local_budget):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try random small step
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    step = local_step * (0.5 + 0.5 * rng.random())
                    candidate = base_x + step * direction
                    candidate = np.clip(candidate, lo, hi)
                    
                    fc = problem(candidate)
                    used += 1
                    
                    # If candidate is better than base in at least one objective and not dominated
                    if np.any(fc < base_f) and not dominates(base_f, fc):
                        base_x = candidate
                        base_f = fc
                        # Update in all_X and all_F
                        all_X[orig_idx] = base_x
                        all_F[orig_idx] = base_f
                        # Also update in F_ref if it's there
                        for k in range(len(F_ref)):
                            if np.allclose(F_ref[k], base_f):
                                F_ref[k] = base_f
                                X_ref[k] = base_x
                                break
                        break  # Move to next non-dominated point after improvement

        return all_X

# EVOLVE-BLOCK-END
