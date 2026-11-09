# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+λ)-ES with adaptive sigma, population-based restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Phase 1: Diverse initialization
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Halton sequence for uniform coverage
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Generate Halton points for initial exploration
        n_halton = 20
        halton_points = np.zeros((n_halton, dim))
        for i in range(1, n_halton + 1):
            for j in range(dim):
                base = primes[j % len(primes)]
                halton_points[i - 1, j] = halton_sequence(i, base)
        halton_points = halton_points * 10.0 - 5.0

        # Strategic points
        strategic_points = [
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.zeros(dim),
        ]

        # Evaluate Halton points
        for pt in halton_points:
            if calls >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Evaluate strategic points
        for pt in strategic_points:
            if calls >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: (1+λ)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 0.8

        # Population size
        lam = 8

        # Track success rate for sigma adaptation
        success_count = 0
        total_count = 0
        adaptation_interval = 10

        # Stagnation tracking
        no_improve_count = 0
        best_f_history = best_f

        # Budget allocation: use last 12% for polish
        polish_start = int(budget * 0.88)

        while calls < budget:
            if time.time() > deadline:
                break

            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.15)
                lam = max(lam // 2, 4)

            # Generate λ offspring
            offspring = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Select best offspring
            offspring.sort(key=lambda t: t[1])
            y_best, fy_best = offspring[0]

            # Count successes
            for _, fy in offspring:
                total_count += 1
                if fy < fx:
                    success_count += 1

            # Update current solution with best offspring
            if fy_best < fx:
                x, fx = y_best, fy_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update global best
            if fy_best < best_f:
                best_f = fy_best
                best_x = y_best.copy()

            # Sigma adaptation every adaptation_interval evaluations
            if total_count >= adaptation_interval:
                success_rate = success_count / total_count
                if success_rate > 0.35:
                    sigma = min(sigma * 1.15, 4.0)
                elif success_rate < 0.15:
                    sigma = max(sigma * 0.75, 0.005)
                else:
                    sigma = min(max(sigma * 1.02, 0.005), 4.0)
                success_count = 0
                total_count = 0

            # Adaptive restart conditions
            progress_ratio = calls / budget
            if in_polish:
                restart_threshold = 20
            else:
                restart_threshold = int(50 * (1.0 - 0.5 * progress_ratio))

            if no_improve_count >= restart_threshold:
                # Population-based restart: mix global and local
                n_restart = 6
                restart_best_x = x.copy()
                restart_best_f = fx

                for ri in range(n_restart):
                    if calls >= budget or time.time() > deadline:
                        break
                    if ri < 3:
                        # Global random
                        cand = rng.uniform(-5, 5, dim)
                    else:
                        # Local perturbation around best
                        local_sigma = max(sigma * 1.5, 0.5)
                        cand = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < restart_best_f:
                        restart_best_f = cand_f
                        restart_best_x = cand.copy()
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = restart_best_x
                fx = restart_best_f
                sigma = max(sigma * 0.9, 0.3)
                no_improve_count = 0
                success_count = 0
                total_count = 0

        # Phase 3: Nelder-Mead polish for final precision
        if calls < budget and time.time() < deadline:
            nm_budget = min(budget - calls, 100)
            if nm_budget >= dim + 2:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, deadline):
        """Nelder-Mead simplex method for final polish."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]
        calls = 1

        # Create initial simplex with small perturbations
        step = 0.05
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < f0:
                f0 = fi
                x0 = xi.copy()

        if calls >= budget:
            return x0, f0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        prev_best = f0

        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if prev_best - fvals[0] < 1e-12:
                # Shrink all points toward best
                for i in range(1, len(simplex)):
                    simplex[i] = np.clip(x0 + sigma_nm * (simplex[i] - x0), -5, 5)
                    fi = func(simplex[i])
                    calls += 1
                    fvals[i] = fi
                    if fi < f0:
                        f0 = fi
                        x0 = simplex[i].copy()
                if calls >= budget:
                    break
                if fvals[0] >= prev_best - 1e-14:
                    break
                prev_best = fvals[0]
                continue

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1
            if calls >= budget:
                break

            if fr < fvals[0]:
                # Try expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if calls >= budget:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < f0:
                        f0 = fe
                        x0 = xe.copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < f0:
                        f0 = fr
                        x0 = xr.copy()
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
                if fr < f0:
                    f0 = fr
                    x0 = xr.copy()
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                fc = func(xc)
                calls += 1
                if calls >= budget:
                    break

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    if fc < f0:
                        f0 = fc
                        x0 = xc.copy()
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(x0 + sigma_nm * (simplex[i] - x0), -5, 5)
                        fi = func(simplex[i])
                        calls += 1
                        fvals[i] = fi
                        if fi < f0:
                            f0 = fi
                            x0 = simplex[i].copy()
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            prev_best = fvals[0]

        return x0, f0
# EVOLVE-BLOCK-END
