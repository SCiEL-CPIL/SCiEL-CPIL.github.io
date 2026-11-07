"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for Katsuura (f23) with hidden shift/rotation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Track calls
        call_count = [0]

        def tracked_func(x):
            if call_count[0] >= budget or time.time() > deadline:
                return np.inf
            call_count[0] += 1
            return func(np.clip(x, -5, 5))

        # Phase 1: Coarse global search with diverse starting points
        n_starts = 40
        pts = []

        # Strategic point: origin
        pts.append(np.zeros(dim))

        # Stratified grid: 2^5 = 32 points covering the box
        for i in range(2 ** dim):
            p = []
            for d in range(dim):
                bit = (i >> d) & 1
                p.append(-5.0 + 10.0 * (bit + 0.5) / 2.0)
            pts.append(np.array(p))

        # Add random points for diversity
        rng = np.random.default_rng(42)
        while len(pts) < n_starts:
            pts.append(rng.uniform(-5, 5, dim))

        # Evaluate all starting points
        evaluated = []
        for p in pts:
            fp = tracked_func(p)
            evaluated.append((fp, p.copy()))

        # Sort by function value
        evaluated.sort(key=lambda x: x[0])

        best_x = evaluated[0][1]
        best_f = evaluated[0][0]

        # Phase 2: Multi-start Nelder-Mead from top-k points
        k = min(5, len(evaluated))

        for i in range(k):
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline:
                break

            x0 = evaluated[i][1]
            remaining = budget - call_count[0]
            if remaining < dim + 2:
                break

            nm_x, nm_f = self._nelder_mead(tracked_func, x0, dim, remaining, deadline, call_count)

            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x

        return best_x, best_f

    def _nelder_mead(self, tracked_func, x0, dim, budget_remaining, deadline, call_count):
        """Nelder-Mead simplex method with time/budget guard."""
        if call_count[0] >= self.budget or time.time() > deadline:
            return x0, np.inf

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [tracked_func(x0)]

        for i in range(1, dim + 1):
            if call_count[0] >= self.budget or time.time() > deadline:
                break
            xi = x0.copy()
            step = 0.5
            xi[i - 1] += step if x0[i - 1] < 0 else -step
            simplex.append(xi)
            fvals.append(tracked_func(xi))

        if call_count[0] >= self.budget or time.time() > deadline:
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        # Nelder-Mead iterations
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        max_iter = 300
        for _ in range(max_iter):
            if call_count[0] >= self.budget or time.time() > deadline:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Convergence check
            if abs(fvals[-1] - fvals[0]) < 1e-12:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = tracked_func(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = tracked_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                fc = tracked_func(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if call_count[0] >= self.budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = tracked_func(simplex[i])
                    if call_count[0] >= self.budget or time.time() > deadline:
                        break

            if call_count[0] >= self.budget or time.time() > deadline:
                break

        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]]
# EVOLVE-BLOCK-END
