"""Rewritten optimizer: global Halton scan then Nelder-Mead refinement."""
import time
import numpy as np


def _halton(index, base):
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    """Global Halton search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin under 300s
        dim = self.dim
        budget = self.budget

        # Split budget: 75% global, 25% local
        n_global = int(budget * 0.75)
        n_local = budget - n_global

        # Global phase: Halton sequence mapped to [-5, 5]^dim
        best_x = None
        best_f = np.inf

        for i in range(1, n_global + 1):
            if time.time() > deadline - 10.0:
                break
            x = np.array([_halton(i + 1, b) for b in range(2, dim + 2)])
            x = -5.0 + 10.0 * x
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Local phase: Nelder-Mead from best point
        if best_x is not None and n_local > 10:
            # Build initial simplex
            simplex = [best_x.copy()]
            for d in range(dim):
                perturbed = best_x.copy()
                step = 0.5 if best_x[d] < 0 else -0.5
                perturbed[d] += step
                simplex.append(perturbed)
            simplex = [np.clip(s, -5, 5) for s in simplex]

            fvals = [func(s) for s in simplex]

            # Track best
            local_best_x = best_x.copy()
            local_best_f = best_f

            # Nelder-Mead iterations
            max_iter = n_local - (dim + 1)
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
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
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])

        return best_x, best_f
