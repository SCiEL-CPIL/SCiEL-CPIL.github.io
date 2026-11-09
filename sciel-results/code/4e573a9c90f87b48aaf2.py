"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a fast (1+1)-ES with adaptive step size and a short
    Nelder-Mead polish phase. Designed for Rastrigin's many local minima:
    the (1+1)-ES quickly finds a good basin, then NM refines.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0

        rng = np.random.default_rng(42)

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Start at origin; Rastrigin global min is often near shifted origin
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        sigma = 2.5  # initial step size, large to explore

        # Use 1/5 success rule for sigma adaptation
        # Target: 1/5 of steps should be successful
        # Simple adaptation: multiply by 1.1 on success, 0.9 on failure
        # But use a more principled approach with a counter

        success_count = 0
        window = 10  # adapt every 10 steps
        step_count = 0

        es_budget = int(budget * 0.75)  # use 75% of budget for ES

        while calls < es_budget:
            if time.time() > deadline - 5:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            step_count += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            else:
                # Keep x unchanged
                pass

            # Adapt sigma every 'window' steps
            if step_count % window == 0:
                success_rate = success_count / window
                # 1/5 rule: if success rate > 0.2, increase sigma; else decrease
                if success_rate > 0.25:
                    sigma *= 1.15
                elif success_rate < 0.15:
                    sigma *= 0.85
                else:
                    sigma *= 1.0
                success_count = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Nelder-Mead local polish from best point
        if calls < budget - 5 and time.time() < deadline - 5:
            x_nm = best_x.copy()

            # Build initial simplex
            simplex = [x_nm.copy()]
            fvals = [best_fx]
            for i in range(dim):
                pert = x_nm.copy()
                # Use a small perturbation in each dimension
                delta = 0.1 if x_nm[i] < 4.9 else -0.1
                pert[i] += delta
                pert = np.clip(pert, -5, 5)
                simplex.append(pert)
                fvals.append(func(pert))
                calls += 1

            # Nelder-Mead iterations
            max_nm_iters = min(60, budget - calls)
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
