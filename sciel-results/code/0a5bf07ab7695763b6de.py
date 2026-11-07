"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer: LHS screening + Nelder-Mead exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Dense Latin Hypercube Sampling for global basin identification
        # We allocate ~20% of budget to screening
        n_lhs = max(10, min(100, budget // 5))
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_lhs equal intervals and pick one point in each
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            intervals = np.arange(n_lhs) / n_lhs
            rng = np.random.default_rng(d + 42)  # deterministic per dim
            offsets = rng.uniform(0, 1, n_lhs)
            lhs_points[:, d] = intervals + offsets
            # Shuffle to maintain LHS property (one per row per column)
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = lhs_points[perm, d]
        
        # Map to [-5, 5]
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate LHS points
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no evals happened
            best_x = np.zeros(dim)
            best_f = func(best_x)
            evals_used = 1

        # Phase 2: Nelder-Mead local optimization from best LHS point
        # Use remaining budget
        remaining_budget = budget - evals_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize simplex
        simplex = []
        fvals = []
        
        # Vertex 0: best point
        simplex.append(best_x.copy())
        fvals.append(best_f)
        
        # Other vertices: perturb along axes
        step_size = 0.5
        for d in range(dim):
            x_new = best_x.copy()
            x_new[d] += step_size
            if x_new[d] > 5.0:
                x_new[d] = 5.0
            simplex.append(x_new)
            fvals.append(func(x_new))
            evals_used += 1
            if time.time() > deadline:
                return best_x, best_f

        fvals = np.array(fvals)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_nm_iter = min(remaining_budget - 1, 2000)
        
        for it in range(max_nm_iter):
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]
            
            # If converged, break
            if fvals[-1] - fvals[0] < 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals_used += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals_used += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals_used += 1
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
