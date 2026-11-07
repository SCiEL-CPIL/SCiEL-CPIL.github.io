"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Stochastic multi-start Nelder-Mead for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Best known value
        best_x = None
        best_f = np.inf

        # Number of restarts
        n_restarts = 10
        budget_per_restart = budget // n_restarts  # 50 calls each

        # Deterministic starting points spread across [-4, 4]^5
        # Use a simple lattice + jitter for diversity
        base_points = []
        # Create a 2x2x2x2x2 grid = 32 points, but we only need 10 distinct ones
        # Let's use a few well-spread deterministic points
        # Point 0: origin
        base_points.append(np.zeros(dim))
        # Point 1: all +3
        base_points.append(np.full(dim, 3.0))
        # Point 2: all -3
        base_points.append(np.full(dim, -3.0))
        # Point 3: alternating +2, -2
        base_points.append(np.array([2.0, -2.0, 2.0, -2.0, 2.0]))
        # Point 4: alternating -2, +2
        base_points.append(np.array([-2.0, 2.0, -2.0, 2.0, -2.0]))
        # Point 5: random-ish deterministic
        base_points.append(np.array([1.5, -1.5, 0.5, -0.5, 2.5]))
        # Point 6
        base_points.append(np.array([-1.5, 1.5, -0.5, 0.5, -2.5]))
        # Point 7
        base_points.append(np.array([3.0, 3.0, -3.0, -3.0, 3.0]))
        # Point 8
        base_points.append(np.array([-3.0, -3.0, 3.0, 3.0, -3.0]))
        # Point 9: small values
        base_points.append(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))

        # Clip to [-4, 4]
        base_points = [np.clip(p, -4, 4) for p in base_points]

        calls_made = 0

        for r in range(n_restarts):
            if calls_made >= budget:
                break

            x_start = base_points[r].copy()
            # Add small jitter to avoid exact duplicates in case of symmetry
            jitter = 0.1 * rng.standard_normal(dim)
            x_start = np.clip(x_start + jitter, -4, 4)

            # Nelder-Mead simplex
            # Initial simplex: base point + small perturbations along each axis
            simplex = [x_start.copy()]
            f_values = [func(x_start)]
            calls_made += 1
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = x_start.copy()

            # Perturbation size
            pert = 0.5
            for i in range(dim):
                v = x_start.copy()
                v[i] += pert
                v = np.clip(v, -5, 5)
                simplex.append(v)
                f_values.append(func(v))
                calls_made += 1
                if f_values[-1] < best_f:
                    best_f = f_values[-1]
                    best_x = v.copy()

            # Nelder-Mead iterations
            max_iters = budget_per_restart - (1 + dim)  # remaining calls for this restart
            for it in range(max_iters):
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < min(fr, f_values[-1]):
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls_made += 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
