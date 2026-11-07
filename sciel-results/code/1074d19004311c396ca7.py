"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Generate Sobol sequence for global exploration
        # Use scipy if available, otherwise use a simple quasi-random approach
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            # Generate points in [0,1]^dim, then map to [-5,5]^dim
            # We want to explore efficiently. Use ~100-200 points for global phase.
            n_global = min(150, budget // 3)
            sobol_samples = sampler.random(n_global)
            x_global = sobol_samples * 10.0 - 5.0
        except ImportError:
            # Fallback: use Halton sequence or stratified sampling
            def halton_sequence(n, dim):
                points = np.zeros((n, dim))
                for d in range(dim):
                    base = d + 2
                    f = np.ones(n)
                    for i in range(1, n):
                        f[i] = 0
                        num = i + 1
                        while num > 0:
                            num, rem = divmod(num, base)
                            f[i] += rem * base ** (-num - 1)
                    points[:, d] = f
                return points

            n_global = min(150, budget // 3)
            x_global = halton_sequence(n_global, dim) * 10.0 - 5.0

        # Phase 1: Global exploration with Sobol points
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_global):
            if time.time() > deadline - 30:
                break
            x = x_global[i]
            fx = func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using Nelder-Mead on top candidates
        # Use remaining budget for local search
        remaining_budget = budget - (budget - 1 - n_global + 1)  # approximate
        remaining_budget = max(budget - n_global - 1, 50)

        # Try Nelder-Mead on the top few candidates
        n_refine = min(5, len(candidates))
        calls_used = n_global

        for idx in range(n_refine):
            if time.time() > deadline - 10:
                break
            if calls_used >= budget - 1:
                break

            fx0, x0 = candidates[idx]

            # Nelder-Mead parameters
            # We'll implement a simple Nelder-Mead
            x = x0.copy()
            fx = fx0

            # Initialize simplex
            simplex = [x.copy()]
            fvals = [fx]

            # Add perturbed points
            for d in range(dim):
                xp = x.copy()
                xp[d] += 0.5 if x[d] < 4.5 else -0.5
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls_used += 1
                simplex.append(xp)
                fvals.append(fp)
                if fp < best_f:
                    best_f = fp
                    best_x = xp.copy()

            # Nelder-Mead iterations
            max_nm_iter = max(20, remaining_budget // (n_refine + 1))

            for nm_iter in range(max_nm_iter):
                if time.time() > deadline - 5:
                    break
                if calls_used >= budget - 1:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_used += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            fvals[i] = fi
                        if calls_used >= budget - 1:
                            break

            # If we still have budget, do a final random perturbation search
            if calls_used < budget - 1 and time.time() < deadline - 5:
                # Quick local random search around best
                sigma = 0.5
                for _ in range(min(50, budget - 1 - calls_used)):
                    if time.time() > deadline - 2:
                        break
                    if calls_used >= budget - 1:
                        break
                    y = np.clip(best_x + sigma * np.random.default_rng(42).standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
