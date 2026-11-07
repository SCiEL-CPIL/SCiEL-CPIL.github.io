"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free optimizer: random restarts + Nelder-Mead simplex."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if np.isfinite(fx):
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find good starting points
        # Use a fraction of budget for global exploration
        n_global = min(50, self.budget // 10)
        rng = np.random.default_rng(42)

        candidates = []
        for _ in range(n_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            fx = safe_call(x)
            if np.isfinite(fx):
                candidates.append((fx, x.copy()))

        if not candidates:
            return best_x, best_f

        # Sort by function value, take top few for local refinement
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead simplex on top candidates
        # Allocate remaining budget among top candidates
        remaining = self.budget - calls_made
        n_local = min(5, len(candidates))
        budget_per_local = max(1, remaining // n_local)

        for i in range(n_local):
            if calls_made >= self.budget or time.time() > deadline:
                break

            x0 = candidates[i][1]
            f0 = candidates[i][0]

            # Build initial simplex
            simplex = [x0.copy()]
            for d in range(self.dim):
                perturbation = x0.copy()
                step = 0.1 * (abs(x0[d]) + 1)
                perturbation[d] += step
                simplex.append(perturbation)

            # Nelder-Mead iterations
            for _ in range(budget_per_local):
                if calls_made >= self.budget or time.time() > deadline:
                    break

                # Evaluate all simplex vertices (cache the first)
                fs = [f0]
                for j in range(1, self.dim + 1):
                    if time.time() > deadline:
                        break
                    fs.append(safe_call(simplex[j]))

                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[j].copy() for j in order]
                fs = [fs[j] for j in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_call(xc)
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, self.dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            fs[j] = safe_call(simplex[j])

                # Update best
                if fs[0] < f0:
                    f0 = fs[0]
                    x0 = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
