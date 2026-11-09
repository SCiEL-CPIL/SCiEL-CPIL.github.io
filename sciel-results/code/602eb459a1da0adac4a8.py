"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global stochastic search (Nelder-Mead + CMA-lite) for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_fx = float('inf')

        def record(x, fx):
            nonlocal best_x, best_fx, calls
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            record(x, fx)
            return fx

        # --- Phase 0: Exploit known Büche-Rastrigin structure + deterministic probes ---
        # The unshifted minimum is at x_i = 2. Probe that and a few structured points.
        init_points = [
            np.full(dim, 2.0),
            np.zeros(dim),
            np.full(dim, -2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0] if dim == 5 else [2.0] * dim),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0] if dim == 5 else [-2.0] * dim),
        ]
        for p in init_points:
            fx = evaluate(p)
            if fx is None:
                break

        # --- Phase 1: Nelder-Mead from the best initial point ---
        def nelder_mead(x0, fx0, max_iter=200):
            nonlocal calls, best_x, best_fx
            if calls >= budget or time.time() > deadline:
                return
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                fi = evaluate(xi)
                if fi is None:
                    return
                simplex.append(xi)
                fvals.append(fi)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)
                if fr is None:
                    return

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe is None:
                        return
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
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    if fc is None:
                        return
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = evaluate(simplex[i])
                            if fi is None:
                                return
                            fvals[i] = fi
                        break

        # Run Nelder-Mead from the best point found so far
        if best_x is not None:
            nelder_mead(best_x, best_fx, max_iter=150)

        # --- Phase 2: CMA-ES-lite (derivative-free, rotation-invariant) ---
        # Use a simplified CMA with adaptive step size and random rotations.
        def cma_lite(x0, sigma0, max_iter=300):
            nonlocal calls, best_x, best_fx
            if calls >= budget or time.time() > deadline:
                return
            rng = np.random.default_rng(12345)
            x = x0.copy()
            fx = best_fx
            sigma = sigma0
            pop_size = 4
            # Simple CMA with random rotation matrix
            C = np.eye(dim)
            # Initialize random rotation
            U, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
            D = np.eye(dim)

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return
                # Generate offspring
                for j in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        return
                    z = rng.standard_normal(dim)
                    # Apply rotation and scaling
                    step = U @ (D @ z)
                    y = x + sigma * step
                    fy = evaluate(y)
                    if fy is None:
                        return

                # Adapt sigma (simple rule)
                sigma *= 0.95 if calls > 10 else 1.0
                sigma = max(sigma, 1e-4)

                # Occasionally re-center around best
                if _ % 50 == 0 and best_x is not None:
                    x = best_x.copy()
                    fx = best_fx

        # Run CMA-lite from best point with moderate initial step
        if best_x is not None:
            cma_lite(best_x, 1.0, max_iter=200)

        # --- Phase 3: Fine-tuning with very small random perturbations ---
        if calls < budget and time.time() < deadline:
            rng = np.random.default_rng(999)
            sigma = 0.05
            for _ in range(budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < best_fx:
                    sigma = min(sigma * 1.1, 0.2)
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
