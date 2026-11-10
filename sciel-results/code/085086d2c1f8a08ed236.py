"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS + Nelder-Mead optimizer for WFG7 with deterministic spread."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limits

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
            if len(F) == 0:
                return np.array([], dtype=int)
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

        # Phase 1: Latin Hypercube Sampling for deterministic space coverage
        n_lhs = min(budget, 80)
        # Generate LHS points: for each dimension, partition into n_lhs equal bins
        # and pick one random point from each bin
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            for i in range(n_lhs):
                bin_idx = perm[i]
                # Random point within bin bin_idx
                frac = (bin_idx + rng.random()) / n_lhs
                lhs_points[i, d] = lo[d] + frac * (hi[d] - lo[d])
        
        F_lhs = np.array([problem(x) for x in lhs_points])
        used = n_lhs

        # Reference point for dominance/hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Phase 2: Select top non-dominated points and apply Nelder-Mead
        # Get non-dominated set from LHS
        nd_idx = non_dominated_set(F_lhs)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])
        
        # Sort by distance to reference point (closer is better)
        nd_points = F_lhs[nd_idx]
        dists = np.sum((nd_points - ref) ** 2, axis=1)
        sorted_nd = nd_idx[np.argsort(dists)]
        
        # Allocate remaining budget for Nelder-Mead on best non-dominated points
        remaining = budget - used
        # Each Nelder-Mead run uses ~2*dim evaluations (initial simplex + iterations)
        # With dim=12, that's ~24 evals per run. Let's do a few runs.
        nm_per_point = 15  # max iterations per NM run
        evals_per_nm = 2 * dim + nm_per_point * dim  # rough estimate
        n_nm_runs = min(len(sorted_nd), max(1, remaining // max(1, evals_per_nm)))
        
        all_X = lhs_points.copy()
        all_F = F_lhs.copy()
        
        for run_i in range(n_nm_runs):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            start_idx = sorted_nd[run_i]
            x0 = all_X[start_idx].copy()
            f0 = all_F[start_idx].copy()
            
            # Define scalarized objective: weighted sum with weights based on distance to ref
            # Use inverse distance to emphasize points closer to ideal
            # For WFG7, use equal weights for simplicity, or adaptive
            weights = np.ones(n_obj) / n_obj
            
            def scalar_obj(x):
                fx = problem(x)
                return np.dot(weights, fx)
            
            # Nelder-Mead implementation
            def nelder_mead(x0, max_iter=15, budget_check=None):
                # Initial simplex
                simplex = [x0.copy()]
                for d in range(dim):
                    x_new = x0.copy()
                    if x0[d] + 0.1 * (hi[d] - lo[d]) <= hi[d]:
                        x_new[d] += 0.1 * (hi[d] - lo[d])
                    else:
                        x_new[d] -= 0.1 * (hi[d] - lo[d])
                    simplex.append(x_new)
                
                simplex_F = [scalar_obj(x) for x in simplex]
                used_count = dim + 1  # initial simplex evaluations
                
                for it in range(max_iter):
                    if budget_check is not None and budget_check[0] >= budget:
                        break
                    if budget_check is not None and time.time() > budget_check[1]:
                        break
                    
                    # Sort simplex by objective value
                    order = np.argsort(simplex_F)
                    simplex = [simplex[i] for i in order]
                    simplex_F = [simplex_F[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    centroid = np.clip(centroid, lo, hi)
                    
                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = scalar_obj(x_ref)
                    used_count += 1
                    if budget_check is not None:
                        budget_check[0] += 1
                        if budget_check[0] >= budget or time.time() > budget_check[1]:
                            break
                    
                    if simplex_F[0] <= f_ref < simplex_F[-2]:
                        simplex[-1] = x_ref
                        simplex_F[-1] = f_ref
                    elif f_ref < simplex_F[0]:
                        # Expansion
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - simplex[-1])
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = scalar_obj(x_exp)
                        used_count += 1
                        if budget_check is not None:
                            budget_check[0] += 1
                            if budget_check[0] >= budget or time.time() > budget_check[1]:
                                break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_F[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_F[-1] = f_ref
                    else:
                        # Contraction
                        rho = 0.5
                        x_con = centroid + rho * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, lo, hi)
                        f_con = scalar_obj(x_con)
                        used_count += 1
                        if budget_check is not None:
                            budget_check[0] += 1
                            if budget_check[0] >= budget or time.time() > budget_check[1]:
                                break
                        if f_con < simplex_F[-1]:
                            simplex[-1] = x_con
                            simplex_F[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lo, hi)
                                simplex_F[i] = scalar_obj(simplex[i])
                                used_count += 1
                                if budget_check is not None:
                                    budget_check[0] += 1
                                    if budget_check[0] >= budget or time.time() > budget_check[1]:
                                        break
                            if budget_check is not None and (budget_check[0] >= budget or time.time() > budget_check[1]):
                                break
                    
                    # Check convergence
                    if abs(simplex_F[0] - simplex_F[-1]) < 1e-6:
                        break
                
                return simplex[0], simplex_F[0]
            
            budget_check = [used, deadline]
            x_best, f_best = nelder_mead(x0, max_iter=nm_per_point, budget_check=budget_check)
            used = budget_check[0]
            
            # Update all_X and all_F with the best point found
            all_X[start_idx] = x_best
            all_F[start_idx] = f0  # Keep original F for dominance, but we want the actual F
            # Actually, we need to get the actual objective values, not scalar
            f_actual = problem(x_best)
            all_F[start_idx] = f_actual
            used += 1  # For the final evaluation to get actual objectives

        # Phase 3: Final non-dominated set extraction
        final_nd = non_dominated_set(all_F)
        
        # Return all evaluated points (the evaluator records all returned objective vectors)
        # But we only return the final population
        return all_X

# EVOLVE-BLOCK-END
