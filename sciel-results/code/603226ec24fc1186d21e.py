import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def _make_func(self, func):
        def wrapped(x):
            self.call_count += 1
            return float(func(x))
        return wrapped

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.call_count = 0

        lb = -5.0
        ub = 5.0
        rng = np.random.default_rng(12345)

        wrapped_func = self._make_func(func)

        # --- Phase 1: Random sampling to get a good initial guess ---
        n_init = 20
        best_x = None
        best_f = np.inf

        # Sample from a few different distributions to cover the space
        # 1. Uniform
        for _ in range(n_init // 2):
            x = rng.uniform(lb, ub, size=dim)
            f = wrapped_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 2. Gaussian centered at 0 with sigma 2
        for _ in range(n_init // 2):
            x = rng.normal(0, 2, size=dim)
            x = np.clip(x, lb, ub)
            f = wrapped_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if self.call_count >= budget:
            return best_x, best_f

        # --- Phase 2: Nelder-Mead local refinement ---
        # Use the best point from Phase 1 as the starting point
        x0 = best_x.copy()
        f0 = best_f

        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            if x0[i] + 0.1 > ub:
                xi[i] = x0[i] - 0.1
            else:
                xi[i] = x0[i] + 0.1
            simplex.append(xi)

        # Evaluate initial simplex
        fvals = [f0]
        for i in range(1, dim + 1):
            f = wrapped_func(simplex[i])
            fvals.append(f)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        max_iter = 100
        for _ in range(max_iter):
            if self.call_count >= budget:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if np.std(fvals) < 1e-10:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:dim], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[dim])
            x_ref = np.clip(x_ref, lb, ub)
            f_ref = wrapped_func(x_ref)
            if self.call_count >= budget:
                break

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, lb, ub)
                f_exp = wrapped_func(x_exp)
                if self.call_count >= budget:
                    break
                if f_exp < f_ref:
                    simplex[dim] = x_exp
                    fvals[dim] = f_exp
                else:
                    simplex[dim] = x_ref
                    fvals[dim] = f_ref
            elif f_ref < fvals[dim - 1]:
                simplex[dim] = x_ref
                fvals[dim] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[dim] - centroid)
                x_con = np.clip(x_con, lb, ub)
                f_con = wrapped_func(x_con)
                if self.call_count >= budget:
                    break
                if f_con < fvals[dim]:
                    simplex[dim] = x_con
                    fvals[dim] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        fvals[i] = wrapped_func(simplex[i])
                        if self.call_count >= budget:
                            break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f
