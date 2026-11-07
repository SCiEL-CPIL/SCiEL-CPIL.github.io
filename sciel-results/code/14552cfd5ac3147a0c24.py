"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Lattice-Local Optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        # Safety margin: 330s kill limit. Use 290s to be safe.
        deadline = start_time + 290.0

        # 1. Deterministic Space-Filling Initialization (Halton Sequence)
        # We want a good spread. Let's use roughly 50% of the budget for initialization.
        # With budget=200, N_init=100.
        N_init = max(10, budget // 2)
        
        # Generate Halton points
        X_init = self._halton_sequence(N_init, dim)
        # Scale to bounds
        X_init = lo + (hi - lo) * X_init
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = N_init
        
        # Check time
        if time.time() > deadline:
            return X_init

        # 2. Identify Non-Dominated Points from Initialization
        # We want to refine the most promising points.
        # Simple approach: Take all non-dominated points.
        # If there are too many, take the ones with the lowest sum of objectives or similar.
        
        # Let's find the non-dominated set
        nd_indices = self._get_non_dominated_indices(F_init)
        
        # If we have too many non-dominated points (e.g. > 10), we need to select a subset for local search
        # to ensure we have budget for the local search steps.
        # Let's say we want to spend the remaining budget on local search.
        # Local search (Nelder-Mead) on 12 vars takes ~100-200 evals per point to converge well.
        # With 100 remaining budget, we can only do ~1-2 points properly.
        # This suggests the "local search on most promising" strategy needs to be very targeted.
        
        # Alternative interpretation of the directive: "iteratively refine the lattice points via local gradient-free optimization ... on the most promising non-dominated points"
        
        # Let's select the top K non-dominated points based on some criterion.
        # Since we don't know the true front, we can use the "closest to reference point" or just the first few.
        # Reference point is [3.0, 5.0, 7.0].
        # Let's score points by distance to a "center" of the front or just pick a diverse subset.
        
        # For simplicity and robustness, let's pick the 2-3 best points based on a simple metric, 
        # or just the first few non-dominated ones if we trust the Halton spread.
        
        # Let's try refining the 3 "best" non-dominated points.
        # "Best" here could mean lowest sum of objectives, or lowest max objective.
        # Let's use the sum of objectives as a heuristic for "closeness to origin" (good for minimization).
        
        if len(nd_indices) == 0:
            # Fallback: take the point with the lowest sum
            idx_min_sum = np.argmin(np.sum(F_init, axis=1))
            nd_indices = [idx_min_sum]
            
        # Sort non-dominated indices by sum of objectives
        sums = F_init[nd_indices].sum(axis=1)
        sorted_nd_indices = nd_indices[np.argsort(sums)]
        
        # How many to refine?
        # Budget remaining: budget - used
        # Nelder-Mead in 12D: initial simplex is 13 points. Each iteration is 1 eval.
        # Convergence might take 50-100 iterations.
        # So 1 point takes ~100 evals.
        # If we have 100 budget left, we can do 1 point.
        # Let's be conservative: refine 1 point if budget allows > 50, else 0.
        # Actually, let's try to refine as many as we can, but cap it.
        
        max_refine = 2
        points_to_refine = sorted_nd_indices[:max_refine]
        
        # Store all results so far
        X_all = [X_init]
        F_all = [F_init]
        
        # 3. Local Refinement (Nelder-Mead)
        for idx in points_to_refine:
            if time.time() > deadline:
                break
            
            x0 = X_init[idx].copy()
            
            # Run Nelder-Mead
            # We need a custom loop to track evaluations and respect deadline/budget.
            x_opt, f_opt, evals_used = self._nelder_mead(problem, x0, lo, hi, budget_remaining=budget - used)
            
            used += evals_used
            
            if evals_used > 0:
                # Add the result to our pool
                # Note: _nelder_mead returns the best point found.
                # We should also potentially keep the path? No, just the best is standard for archive.
                # But the task says "Every returned objective vector is recorded".
                # The `problem` function is called inside _nelder_mead.
                # The evaluator records what `problem` returns.
                # So we just need to ensure `problem` is called.
                
                # We don't strictly need to append to X_all/F_all for the return value, 
                # because the evaluator records the calls to `problem` directly?
                # "Every returned objective vector is recorded and the return value of __call__ is ignored."
                # This implies the evaluator wraps `problem` or `__call__`?
                # "evaluator calls ... and calls it once."
                # "Every returned objective vector is recorded" -> This likely refers to the output of `problem(x)`.
                # So the history of calls to `problem` is the archive.
                # Therefore, I just need to make sure I call `problem` within the budget.
                
                pass # No need to update local arrays for return value, as return is ignored.

        # The return value is ignored, but we must return something.
        # Let's return the initial population or the last refined point.
        # It doesn't matter for scoring, but must be valid.
        return X_init

    def _halton_sequence(self, N, dim):
        """Generate N points in [0, 1)^dim using the Halton sequence."""
        points = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                points[i, d] = self._halton_value(i + 1, d + 2) # Base starts at 2
        return points

    def _halton_value(self, n, base):
        """Compute the n-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        return r

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = F.shape[0]
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd

    def _nelder_mead(self, problem, x0, lo, hi, budget_remaining=50):
        """
        Simple Nelder-Mead implementation with budget and time constraints.
        Returns (best_x, best_f, evals_used).
        """
        dim = len(x0)
        # Initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            # Perturb one dimension
            step = 0.05 * (hi[i-1] - lo[i-1]) if (hi[i-1] - lo[i-1]) > 0 else 0.05
            simplex[i, i-1] += step
        
        # Evaluate initial simplex
        F_simplex = np.zeros(dim + 1)
        evals = 0
        for i in range(dim + 1):
            x = np.clip(simplex[i], lo, hi)
            F_simplex[i] = np.sum(problem(x)) # Use sum as scalar objective for NM
            evals += 1
            if evals >= budget_remaining:
                break
        
        if evals >= budget_remaining:
            best_idx = np.argmin(F_simplex)
            return simplex[best_idx], F_simplex[best_idx], evals

        # Main loop
        max_iter = budget_remaining - evals
        for _ in range(max_iter):
            # Sort simplex by objective value
            sort_idx = np.argsort(F_simplex)
            simplex = simplex[sort_idx]
            F_simplex = F_simplex[sort_idx]
            
            # Centroid of all but the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = np.sum(problem(x_ref))
            evals += 1
            if evals >= budget_remaining:
                break
            
            # Expansion
            if f_ref < F_simplex[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = np.sum(problem(x_exp))
                evals += 1
                if evals >= budget_remaining:
                    break
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    F_simplex[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    F_simplex[-1] = f_ref
            # Contraction
            elif f_ref < F_simplex[-2]:
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = np.sum(problem(x_con))
                evals += 1
                if evals >= budget_remaining:
                    break
                
                if f_con <= f_ref:
                    simplex[-1] = x_con
                    F_simplex[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        x_shr = np.clip(simplex[i], lo, hi)
                        F_simplex[i] = np.sum(problem(x_shr))
                        evals += 1
                        if evals >= budget_remaining:
                            break
                    if evals >= budget_remaining:
                        break
            # Shrink
            else:
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    x_shr = np.clip(simplex[i], lo, hi)
                    F_simplex[i] = np.sum(problem(x_shr))
                    evals += 1
                    if evals >= budget_remaining:
                        break
                if evals >= budget_remaining:
                    break
            
            # Check convergence (simple check: range of F is small)
            if np.ptp(F_simplex) < 1e-6:
                break

        best_idx = np.argmin(F_simplex)
        return simplex[best_idx], F_simplex[best_idx], evals
# EVOLVE-BLOCK-END
