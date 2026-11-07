"""Optimizer for BBOB-style f6 (Attractive sector).

Strategy:
  1. Fast global scatter (cheap early progress for AOCC).
  2. Nelder-Mead simplex refinement from the best point (exploits smooth structure).
  3. Multi-start restarts from scattered points if the budget/time allow.
  4. Explicit deadline guard so we always return the best found.
"""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with a global scatter and a time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, dim, rng, deadline, calls_state):
        """Nelder-Mead refinement starting from x0.

        Returns (best_x, best_fx, calls_used).
        calls_state is a one-element list [count] shared with the caller.
        """
        # Build initial simplex
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls_state[0] += 1
        best_x, best_fx = x.copy(), fx

        simplex = [x.copy()]
        simplex_fx = [fx]

        # Add dim more points, each a perturbation of x0
        base_step = 0.35
        for i in range(dim):
            xi = x.copy()
            xi[i] += base_step
            xi = np.clip(xi, -5.0, 5.0)
            fxi = func(xi)
            calls_state[0] += 1
            simplex.append(xi)
            simplex_fx.append(fxi)
            if fxi < best_fx:
                best_x, best_fx = xi.copy(), fxi

        alpha = 1.0     # reflection
        gamma = 2.0     # expansion
        rho = 0.5       # contraction
        sigma = 0.5     # shrink
        max_iter = 600
        tol = 1e-10

        for _ in range(max_iter):
            if calls_state[0] >= budget or time.time() > deadline:
                break

            # Order simplex by fitness (best first)
            order = np.argsort(simplex_fx)
            simplex = [simplex[i] for i in order]
            simplex_fx = [simplex_fx[i] for i in order]

            # Convergence check
            if abs(simplex_fx[0] - simplex_fx[-1]) < tol:
                break

            # Centroid of all but worst
            worst = simplex[-1]
            worst_fx = simplex_fx[-1]
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            calls_state[0] += 1
            if fr < best_fx:
                best_x, best_fx = xr.copy(), fr

            if fr < simplex_fx[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                calls_state[0] += 1
                if fe < best_fx:
                    best_x, best_fx = xe.copy(), fe
                if fe < fr:
                    simplex[-1] = xe
                    simplex_fx[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_fx[-1] = fr
            elif fr < simplex_fx[-2]:
                # Simple reflection
                simplex[-1] = xr
                simplex_fx[-1] = fr
            else:
                # Contraction
                if fr < worst_fx:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (worst - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                calls_state[0] += 1
                if fc < best_fx:
                    best_x, best_fx = xc.copy(), fc
                if fc < min(fr, worst_fx):
                    simplex[-1] = xc
                    simplex_fx[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = func(simplex[i])
                        calls_state[0] += 1
                        simplex_fx[i] = fi
                        if fi < best_fx:
                            best_x, best_fx = simplex[i].copy(), fi

        return best_x, best_fx, calls_state[0]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_state = [0]

        # ---- Phase 1: global scatter for fast early progress ----
        # Spread points over the box so we quickly find a decent region.
        n_scatter = 20
        scatter_points = []
        for i in range(n_scatter):
            if calls_state[0] >= budget or time.time() > deadline:
                break
            # Mix of a coarse grid and random
            if i < 9:
                # 3x3 grid in 2D-ish, but fill 5D with structured offsets
                # Use a simple LHS-like spread via deterministic + random
                pt = np.array([
                    -4.0 + 8.0 * ((i * 7 + d * 3) % 9) / 8.0
                    for d in range(dim)
                ])
            else:
                pt = rng.uniform(-4.5, 4.5, dim)
            pt = np.clip(pt, -5.0, 5.0)
            f = func(pt)
            calls_state[0] += 1
            scatter_points.append((pt, f))
            if f < best_fx:
                best_x, best_fx = pt.copy(), f

        # ---- Phase 2: Nelder-Mead refinement from the best point ----
        if best_x is not None and calls_state[0] < budget and time.time() < deadline:
            bx, bf, _ = self._nelder_mead(
                func, best_x, budget, dim, rng, deadline, calls_state
            )
            if bf < best_fx:
                best_x, best_fx = bx, bf

        # ---- Phase 3: multi-start from other scatter points if budget allows ----
        # Sort scatter points by fitness (best first), skip the one already refined.
        scatter_points.sort(key=lambda t: t[1])
        for pt, f in scatter_points:
            if calls_state[0] >= budget or time.time() > deadline:
                break
            # Only refine points that are reasonably close to the best
            if f > best_fx + 5.0:
                continue
            # Skip the point that is essentially the current best
            if np.allclose(pt, best_x, atol=1e-6):
                continue
            bx, bf, _ = self._nelder_mead(
                func, pt, budget, dim, rng, deadline, calls_state
            )
            if bf < best_fx:
                best_x, best_fx = bx, bf

        # ---- Phase 4: final fine polish near the global best if budget remains ----
        if best_x is not None and calls_state[0] < budget and time.time() < deadline:
            bx, bf, _ = self._nelder_mead(
                func, best_x, budget, dim, rng, deadline, calls_state
            )
            if bf < best_fx:
                best_x, best_fx = bx, bf

        return best_x, best_fx
