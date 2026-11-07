"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s

        # Deterministic starting points spread in [-4, 4]^5
        points = []
        points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                points.append(p)
        for sign in [1, -1]:
            points.append(np.full(dim, sign * 2.0))
        
        start_points = points[:10]
        start_points = sorted(start_points, key=lambda p: np.linalg.norm(p))

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_eval(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget or time.time() > deadline:
                return None
            total_calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        for start in start_points:
            if total_calls >= budget or time.time() > deadline:
                break

            # Nelder-Mead simplex
            x0 = start.copy()
            f0 = safe_eval(x0)
            if f0 is None:
                break

            # Build simplex
            simplex = [x0.copy()]
            fvals = [f0]
            valid_simplex = True
            for i in range(dim):
                xi = x0.copy()
                # Use a smaller step to stay within bounds and be more local
                step = 0.1
                xi[i] += step * (1 if xi[i] < 5 else -1)
                xi = np.clip(xi, -5, 5)
                fi = safe_eval(xi)
                if fi is None:
                    valid_simplex = False
                    break
                simplex.append(xi)
                fvals.append(fi)
            
            if not valid_simplex:
                continue

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Limit iterations per run to prevent single run from eating all budget
            # 10 runs * 40 iterations = 400 calls approx + 10*6 init = 460 calls
            max_iters_per_run = 30 
            for _ in range(max_iters_per_run):
                if total_calls >= budget or time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_eval(xr)
                if fr is None:
                    break

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_eval(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = safe_eval(xc)
                    if fc is None:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        shrink_ok = True
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_eval(simplex[i])
                            if fi is None:
                                shrink_ok = False
                                break
                            fvals[i] = fi
                        if not shrink_ok:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
