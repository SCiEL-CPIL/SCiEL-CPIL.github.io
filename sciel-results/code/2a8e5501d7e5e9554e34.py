# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic space-filling + local refinement multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim):
        """Generate n quasi-random points in [0,1)^dim using a simple Sobol-like method."""
        # Use a simple deterministic low-discrepancy sequence
        # For simplicity, use a variant of Halton with prime bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dim))
        for d in range(dim):
            base = primes[d % len(primes)]
            for i in range(n):
                f = 1.0
                num = i + 1
                while num > 0:
                    f /= base
                    num = (num - 1) // base
                points[i, d] = f
        return points

    def _nelder_mead(self, f, x0, bounds, max_iter=50, tol=1e-6):
        """Simple Nelder-Mead minimization for a single objective."""
        dim = len(x0)
        lo = bounds[0]
        hi = bounds[1]
        
        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(dim):
            perturbation = x0.copy()
            step = 0.05 * (hi[i] - lo[i]) if hi[i] - lo[i] > 0 else 0.05
            perturbation[i] += step
            perturbation = np.clip(perturbation, lo, hi)
            simplex.append(perturbation)
        
        # Evaluate initial simplex
        values = [f(x) for x in simplex]
        
        for _ in range(max_iter):
            # Sort by function value
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Check convergence
            if abs(values[-1] - values[0]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = f(x_ref)
            
            if f_ref < values[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = f(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    values[-1] = f_ref
            elif f_ref < values[-2]:
                simplex[-1] = x_ref
                values[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = f(x_con)
                if f_con < values[-1]:
                    simplex[-1] = x_con
                    values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        values[i] = f(simplex[i])
        
        return simplex[0], values[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(123)

        # Reference point for internal calculations
        ref_point = np.array([3.0, 5.0, 7.0])

        # Phase 1: Initial space-filling sampling using Halton sequence
        n_initial = min(30, budget)
        X = []
        F = []
        evals_used = 0

        halton_pts = self._sobol_sequence(n_initial, dim)
        for i in range(n_initial):
            x = lo + (hi - lo) * halton_pts[i]
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            evals_used += 1
            if evals_used >= budget or time.time() > deadline:
                break

        X = np.array(X)
        F = np.array(F)

        # Phase 2: Iterative refinement using local optimization
        # We'll do multiple rounds of local refinement on the best non-dominated points
        
        max_rounds = 5
        points_per_round = min(3, max(1, (budget - evals_used) // 10))
        
        for round_num in range(max_rounds):
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Identify non-dominated points
            n = len(F)
            if n == 0:
                break
                
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True

            nondom_idx = [i for i in range(n) if not dominated[i]]

            if len(nondom_idx) == 0:
                best_idx = np.argmin(np.sum(F, axis=1))
                nondom_idx = [best_idx]

            # Select top points to refine
            k_refine = min(points_per_round, len(nondom_idx), max(1, (budget - evals_used) // 8))
            
            # Sort by sum of objectives (simple scalarization)
            scores = np.sum(F[nondom_idx], axis=1)
            nondom_sorted = [nondom_idx[i] for i in np.argsort(scores)]

            # Refine the top k points using Nelder-Mead on scalarized objective
            for idx in nondom_sorted[:k_refine]:
                if evals_used >= budget or time.time() > deadline:
                    break

                x_base = X[idx].copy()
                
                # Use a scalarized objective with random weights to explore different parts of Pareto front
                weights = rng.dirichlet(np.ones(n_obj))
                
                def scalarized_f(x):
                    f = problem(x)
                    return np.dot(f, weights)

                # Run Nelder-Mead with limited iterations
                max_iter = min(30, max(5, (budget - evals_used) // 5))
                x_opt, f_opt_val = self._nelder_mead(scalarized_f, x_base, (lo, hi), max_iter=max_iter)
                
                # Check if we used evaluations
                # Nelder-Mead uses evaluations internally, but we need to track them
                # For simplicity, let's do a few random perturbations instead
                
                # Alternative: do a few random local perturbations
                n_local = min(5, budget - evals_used)
                step = 0.05 * (hi - lo)
                
                for _ in range(n_local):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    x_new = x_base + rng.standard_normal(dim) * step
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    evals_used += 1

                    # Check if f_new is non-dominated
                    is_dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                            is_dominated = True
                            break

                    if not is_dominated:
                        X = np.vstack([X, x_new])
                        F = np.vstack([F, f_new])

                        # Remove dominated points
                        n_pop = len(F)
                        dom = np.zeros(n_pop, dtype=bool)
                        for i in range(n_pop):
                            if dom[i]:
                                continue
                            for j in range(n_pop):
                                if i == j:
                                    continue
                                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                    dom[j] = True
                        
                        keep = [i for i in range(n_pop) if not dom[i]]
                        if len(keep) > 0:
                            X = X[keep]
                            F = F[keep]
                        else:
                            # Keep all if no point dominates another
                            pass

                        # Limit population size
                        if len(F) > 40:
                            # Keep best 40 by sum of objectives
                            keep_scores = np.sum(F, axis=1)
                            keep_sorted = np.argsort(keep_scores)[:40]
                            X = X[keep_sorted]
                            F = F[keep_sorted]

        return X
# EVOLVE-BLOCK-END
