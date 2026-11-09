"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Grid Search with Adaptive Refinement for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def check_time():
            return time.time() < deadline - 5

        def evaluate_grid(center, half_width, n_per_dim, budget_remaining, func, best_x, best_f):
            """Evaluate a grid around center with n_per_dim points per dimension."""
            if not check_time() or budget_remaining <= 0:
                return best_x, best_f, 0

            # For high dimensions, we can't do a full grid. Use coordinate descent style.
            # For dim=5, a full grid with n_per_dim=3 gives 3^5=243 points.
            # With n_per_dim=2, gives 2^5=32 points.
            
            if n_per_dim ** dim > budget_remaining:
                # Can't afford full grid, use coordinate-wise search
                return coordinate_search(center, half_width, budget_remaining, func, best_x, best_f)
            
            # Full grid search
            points = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        for l in range(n_per_dim):
                            for m in range(n_per_dim):
                                pts = np.array([
                                    center[0] + (i - (n_per_dim-1)/2) * (2*half_width/(n_per_dim-1)) if n_per_dim > 1 else center[0],
                                    center[1] + (j - (n_per_dim-1)/2) * (2*half_width/(n_per_dim-1)) if n_per_dim > 1 else center[1],
                                    center[2] + (k - (n_per_dim-1)/2) * (2*half_width/(n_per_dim-1)) if n_per_dim > 1 else center[2],
                                    center[3] + (l - (n_per_dim-1)/2) * (2*half_width/(n_per_dim-1)) if n_per_dim > 1 else center[3],
                                    center[4] + (m - (n_per_dim-1)/2) * (2*half_width/(n_per_dim-1)) if n_per_dim > 1 else center[4],
                                ])
                                points.append(pts)
            
            evals = 0
            for pt in points:
                if not check_time() or budget_remaining - evals <= 0:
                    break
                pt = np.clip(pt, -5, 5)
                f = func(pt)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()
            
            return best_x, best_f, evals

        def coordinate_search(center, half_width, budget_remaining, func, best_x, best_f):
            """Coordinate-wise search: try perturbations along each axis."""
            if not check_time() or budget_remaining <= 0:
                return best_x, best_f, 0
            
            evals = 0
            # Try several step sizes
            for frac in [1.0, 0.5, 0.25, 0.125]:
                if not check_time() or budget_remaining - evals <= 0:
                    break
                step = half_width * frac
                for d in range(dim):
                    for direction in [1, -1]:
                        if not check_time() or budget_remaining - evals <= 0:
                            break
                        pt = center.copy()
                        pt[d] += direction * step
                        pt = np.clip(pt, -5, 5)
                        f = func(pt)
                        evals += 1
                        if f < best_f:
                            best_f = f
                            best_x = pt.copy()
                            center = pt.copy()
            
            return best_x, best_f, evals

        # Phase 1: Coarse grid to find approximate basin
        # Start with a coarse grid over [-5, 5]^5
        # 3^5 = 243 evaluations for a 3x3x3x3x3 grid
        best_x = np.array([0.0] * dim)
        best_f = func(best_x)
        evals_used = 1

        # Coarse grid: 3 points per dimension = 243 evals
        if check_time() and evals_used < budget - 100:
            bx, bf, used = evaluate_grid(
                np.array([0.0] * dim), 
                5.0, 
                3, 
                budget - evals_used - 50, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        # Phase 2: Medium grid around best point
        if check_time() and evals_used < budget - 50:
            half_width = 2.0
            bx, bf, used = evaluate_grid(
                best_x, 
                half_width, 
                3, 
                budget - evals_used - 30, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        # Phase 3: Fine grid
        if check_time() and evals_used < budget - 20:
            half_width = 0.8
            bx, bf, used = evaluate_grid(
                best_x, 
                half_width, 
                3, 
                budget - evals_used - 15, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        # Phase 4: Very fine grid
        if check_time() and evals_used < budget - 10:
            half_width = 0.3
            bx, bf, used = evaluate_grid(
                best_x, 
                half_width, 
                3, 
                budget - evals_used - 5, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        # Phase 5: Ultra-fine coordinate search for final refinement
        if check_time() and evals_used < budget:
            bx, bf, used = coordinate_search(
                best_x, 
                0.15, 
                budget - evals_used, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        # Phase 6: Even finer if we have budget left
        if check_time() and evals_used < budget:
            bx, bf, used = coordinate_search(
                best_x, 
                0.05, 
                budget - evals_used, 
                func, 
                best_x, 
                best_f
            )
            best_x, best_f = bx, bf
            evals_used += used

        return best_x, best_f
# EVOLVE-BLOCK-END
