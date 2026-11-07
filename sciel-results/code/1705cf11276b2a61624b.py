# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based decomposition with local Nelder-Mead simplex per cell."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []
        eval_count = [0]

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def count_eval(x):
            if eval_count[0] >= budget:
                raise StopIteration
            eval_count[0] += 1
            return eval_point(x)

        # Phase 1: Grid decomposition with 2x2x2x2 = 16 cells
        grid_size = 2
        n_cells = grid_size ** dim

        # Generate 16 weight vectors for scalarization (2 objectives)
        weights = np.linspace(0.0, 1.0, n_cells)

        cell_bounds = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        lo_c = lo + span * np.array([i, j, k, l]) / grid_size
                        hi_c = lo + span * np.array([i + 1, j + 1, k + 1, l + 1]) / grid_size
                        cell_bounds.append((lo_c, hi_c))

        # Allocate budget: 16 cells, ~12 evals each = 192, leaving 8
        evals_per_cell = 12

        for idx, (lo_c, hi_c) in enumerate(cell_bounds):
            if eval_count[0] >= budget:
                break

            w = weights[idx]

            def scalar_obj(x, w=w):
                f = count_eval(x)
                return w * f[0] + (1 - w) * f[1]

            x0 = (lo_c + hi_c) / 2.0

            try:
                maxfev = min(evals_per_cell, budget - eval_count[0])
                result = minimize(
                    scalar_obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'maxfev': maxfev,
                        'xatol': 1e-5,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except (StopIteration, Exception):
                pass

        # Phase 2: Use remaining budget for additional scalarizations at global scale
        if eval_count[0] < budget:
            remaining = budget - eval_count[0]
            # Use intermediate weights that weren't covered
            extra_weights = [0.125, 0.375, 0.625, 0.875]
            for w_extra in extra_weights:
                if eval_count[0] >= budget:
                    break
                if budget - eval_count[0] < dim + 2:
                    break

                def scalar_obj_extra(x, w=w_extra):
                    f = count_eval(x)
                    return w * f[0] + (1 - w) * f[1]

                x0_global = (lo + hi) / 2.0
                try:
                    maxfev_extra = min(budget - eval_count[0], 8)
                    result = minimize(
                        scalar_obj_extra,
                        x0_global,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 20,
                            'maxfev': maxfev_extra,
                            'xatol': 1e-5,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except (StopIteration, Exception):
                    pass

        # Phase 3: Fill any remaining budget with targeted evaluations
        while eval_count[0] < budget:
            # Try a few more weight values at different locations
            w_fill = 0.5
            x_fill = (lo + hi) / 2.0 + 0.05 * span * np.random.randn(dim)
            x_fill = np.clip(x_fill, lo, hi)
            try:
                f = count_eval(x_fill)
            except (StopIteration, Exception):
                break

        return all_X
# EVOLVE-BLOCK-END
