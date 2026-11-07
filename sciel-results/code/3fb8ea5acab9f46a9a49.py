"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start local search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # 1. Initial sampling: 40 points to cover the space
        init_n = 40
        rng = np.random.default_rng(42)
        X_init = lo + (hi - lo) * rng.random((init_n, dim))
        F_init = np.array([problem(x) for x in X_init])
        X = X_init.copy()
        F = F_init.copy()
        used = init_n

        # 2. Local Search Phase: Nelder-Mead on non-dominated points
        # We select non-dominated points and run a short NM on each
        # Objective: weighted sum of objectives. Weights are chosen to target different parts of the front.
        
        # Identify non-dominated points
        def is_dominated(f, F_set):
            return np.any(np.all(F_set <= f, axis=1) & np.any(F_set < f, axis=1))
        
        nd_mask = np.array([not is_dominated(f, F) for f in F])
        nd_indices = np.where(nd_mask)[0]
        
        # If no ND points (shouldn't happen with random init usually, but safe), use all
        if len(nd_indices) == 0:
            nd_indices = np.arange(len(F))
            
        # We will run NM on a subset of these ND points.
        # To ensure we explore the whole front, we can assign different weights to different ND points.
        # However, a simpler robust strategy for WFG9 (which is separable-ish in difficulty) is to just run NM with 
        # a few different weight vectors on the best candidates.
        
        # Let's run NM on up to 10 of the ND points, using different weight vectors.
        # Weight vectors: uniform spread on the simplex.
        n_starts = min(10, len(nd_indices))
        
        # Generate weight vectors
        # For 3 objectives, we can use:
        # (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1/3, 1/3, 1/3)
        # But we have 10 starts. Let's just use a fixed set of 10 diverse weights.
        weights = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7]
        ])
        
        # Shuffle ND indices to pick a diverse set? Or just take the first N?
        # Taking the first N ND points might be biased if they are clustered.
        # Let's pick ND points that are spread out in objective space? 
        # For simplicity and speed, let's just take the first 10 ND points.
        
        for i in range(n_starts):
            if used >= self.budget:
                break
            if time.time() > deadline - 5: # Safety margin
                break
                
            start_idx = nd_indices[i]
            x0 = X[start_idx].copy()
            w = weights[i % len(weights)]
            
            # Nelder-Mead implementation
            # Initial simplex: x0 and x0 + eps * e_j
            eps = 0.05 * (hi - lo)
            simplex = np.vstack([x0])
            for j in range(dim):
                x_j = x0.copy()
                x_j[j] += eps[j]
                simplex = np.vstack([simplex, x_j])
            
            # Evaluate simplex
            F_simplex = np.array([problem(x) for x in simplex])
            used += dim + 1
            
            # Objective function for NM
            def obj(x):
                fx = problem(x)
                nonlocal used
                used += 1
                return np.dot(fx, w)
            
            # We already evaluated the simplex points, so we can reuse them.
            # But NM needs to manage the evaluations. 
            # To strictly adhere to budget, we must count every problem(x) call.
            # Let's implement a simple NM loop that uses the pre-evaluated points.
            
            # Sort simplex by objective value
            order = np.argsort([np.dot(f, w) for f in F_simplex])
            simplex = simplex[order]
            F_simplex = F_simplex[order]
            
            # NM Iterations
            # Max iterations per start: limited by remaining budget
            max_iter = 20
            for _ in range(max_iter):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = F_simplex[-1]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid - alpha * (x_worst - centroid)
                x_r = np.clip(x_r, lo, hi)
                f_r = obj(x_r)
                
                if f_r < np.dot(F_simplex[0], w): # Better than best
                    # Expansion
                    gamma = 2.0
                    x_e = centroid - gamma * (x_worst - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = obj(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        F_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        F_simplex[-1] = f_r
                elif f_r < np.dot(F_simplex[-2], w): # Better than 2nd worst
                    simplex[-1] = x_r
                    F_simplex[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    x_c = centroid + beta * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = obj(x_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        F_simplex[-1] = f_c
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            F_simplex[k] = obj(simplex[k])
                
                # Re-sort simplex
                order = np.argsort([np.dot(f, w) for f in F_simplex])
                simplex = simplex[order]
                F_simplex = F_simplex[order]
                
                # Convergence check (simple: if simplex is small)
                if np.max(np.abs(simplex - simplex[0])) < 1e-6:
                    break
            
            # Record the best point found in this NM run
            best_idx = 0
            best_x = simplex[best_idx]
            best_f = F_simplex[best_idx]
            
            # Add to main population
            X = np.vstack([X, best_x])
            F = np.vstack([F, best_f])

        # 3. Final Polish: If budget remains, run NM on the global best (lowest sum) with a tighter tolerance?
        # Or just return.
        
        return X
# EVOLVE-BLOCK-END
