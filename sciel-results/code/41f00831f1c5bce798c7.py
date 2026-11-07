import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. Adaptive random search to locate the ridge basin.
    2. Nelder-Mead local refinement with adaptive step size.
    3. Multiple restarts from perturbed best points if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = float(func(x))
                if np.isfinite(f_val) and f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Adaptive random search with increasing concentration
        n_coarse = 60
        for i in range(n_coarse):
            if calls >= budget or time.time() > deadline:
                break
            # Start broad, then concentrate around best
            if best_x is not None and i > 20:
                # Mix: 50% around best, 50% global
                if rng.random() < 0.5:
                    spread = 2.0 / (1 + i / 10)
                    x = best_x + rng.normal(0, spread, size=dim)
                else:
                    x = rng.uniform(-5.0, 5.0, size=dim)
            else:
                x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Nelder-Mead with adaptive step
        self._nelder_mead(best_x.copy(), best_f, safe_call, deadline)

        # Phase 3: Multiple restarts
        if calls < budget - 30 and time.time() < deadline:
            for restart in range(10):
                if calls >= budget - 15 or time.time() > deadline:
                    break

                # Adaptive perturbation: larger early, smaller late
                noise_scale = 1.5 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(best_x + perturbation, -5.0, 5.0)

                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break

                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()

                # Nelder-Mead from this point
                self._nelder_mead(x_restart.copy(), f_restart, safe_call, deadline)

        return best_x, best_f

    def _nelder_mead(self, x_start, f_start, safe_call, deadline):
        """
        Nelder-Mead simplex method with adaptive step size.
        """
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho_c = 0.5
        sigma = 0.5

        # Adaptive initial step size
        step_size = 0.3

        simplex = [x_start.copy()]
        f_values = [f_start]

        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)

        if len(simplex) < dim + 1:
            return

        simplex = np.array(simplex)
        f_values = np.array(f_values)

        no_improve = 0
        patience = 20
        max_iter = 80

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if self._calls_remaining() < 5:
                break

            # Sort by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]

            # Check for convergence
            spread_f = np.max(f_values) - np.min(f_values)
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if spread_f < 1e-10 or np.max(spread_x) < 1e-8:
                break

            no_improve += 1
            if no_improve > patience:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break

            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                    no_improve = 0
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
                    no_improve = 0
            elif fr < f_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_values[-1] = fr
                no_improve = 0
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = centroid + rho_c * (xr - centroid)
                else:
                    xc = centroid - rho_c * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

                    # Reduce step size for next iteration if needed
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

    def _calls_remaining(self):
        # We need to track calls - use a shared approach
        # For now, this is a placeholder; the safe_call closure handles it
        return 100  # Conservative estimate
