"""Optimizer using CMA-ES with adaptive local polishing and restarts for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive local search and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin from 330s limit
        n = self.dim
        budget = self.budget

        # Global state
        best_x = None
        best_fx = np.inf
        total_calls = 0

        # CMA-ES Parameters tuned for low dimensions (n=5)
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        rng = np.random.default_rng(42)

        def nelder_mead(x0, max_evals, current_sigma):
            """Nelder-Mead local polishing with adaptive step size."""
            simplex = [x0.copy() for _ in range(n + 1)]
            # Scale step by current sigma but ensure minimum step
            step = max(0.05, 0.3 * current_sigma)
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += step
            
            fvals = np.array([func(x) for x in simplex])
            local_evals = n + 1

            # Track best found during NM
            best_local_x = x0.copy()
            best_local_f = fvals[0]

            for _ in range(max_evals // (n + 1) + 1):
                if time.time() > deadline - 5:
                    break
                if local_evals >= max_evals:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Update best
                if fvals[0] < best_local_f:
                    best_local_f = fvals[0]
                    best_local_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                local_evals += 1

                if fr < best_local_f:
                    best_local_f = fr
                    best_local_x = xr.copy()

                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    local_evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_local_f:
                            best_local_f = fe
                            best_local_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    local_evals += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_local_f:
                            best_local_f = fc
                            best_local_x = xc.copy()
                    else:
                        # Shrink simplex
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            local_evals += 1
                            if fvals[i] < best_local_f:
                                best_local_f = fvals[i]
                                best_local_x = simplex[i].copy()
                        if local_evals >= max_evals:
                            break

            return best_local_x, best_local_f, local_evals

        # Final high-precision polish pass
        def final_polish(x0, remaining_budget):
            """High-precision Nelder-Mead with tight tolerance."""
            if remaining_budget < 10:
                return x0.copy(), func(x0)
            
            x0 = x0.copy()
            f0 = func(x0)
            total_calls_local = 1
            best_x = x0.copy()
            best_f = f0
            
            # Use very small initial step for precision
            step = 0.01
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += step
            
            fvals = np.array([func(x) for x in simplex])
            total_calls_local += n + 1
            
            for _ in range(remaining_budget // (n + 1)):
                if total_calls_local >= remaining_budget:
                    break
                if time.time() > deadline - 2:
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]
                
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                total_calls_local += 1
                
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    total_calls_local += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    total_calls_local += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            total_calls_local += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if total_calls_local >= remaining_budget:
                            break
                
                # Adapt step size based on progress
                if fvals[0] < fvals[-1] * 0.99:
                    step *= 0.7
                else:
                    step = max(0.001, step * 0.8)
            
            return best_x, best_f, total_calls_local

        # Main Loop with Restart Capability
        restart_count = 0
        max_restarts = 8  # Increased restarts for better exploration
        local_search_budget = 25
        stagnation_threshold = 6  # More aggressive restart trigger

        while total_calls < budget and restart_count <= max_restarts:
            if time.time() > deadline - 15:
                break

            # Initialize CMA-ES state
            if best_x is not None and restart_count > 0:
                # For restarts: use best_x with random perturbation to escape local minima
                x = best_x.copy()
                perturbation_scale = 1.5 / (1 + restart_count * 0.5)
                x += rng.standard_normal(n) * perturbation_scale
                x = np.clip(x, -5, 5)
                sigma = max(0.3, 0.8 / (1 + restart_count * 0.3))
            elif best_x is not None:
                x = best_x.copy()
                sigma = 0.8
            else:
                x = np.zeros(n)
                sigma = 1.2

            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            no_improve = 0
            generation = 0
            restart_needed = False

            # Run CMA-ES until budget is exhausted or stagnation triggers restart
            while total_calls < budget:
                if time.time() > deadline - 10:
                    break

                # Generate population
                pops = []
                fits = []

                for i in range(lambda_pop):
                    if total_calls >= budget or time.time() > deadline - 5:
                        break
                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                        y = np.dot(L, z)
                    except:
                        y = z
                    x_i = np.clip(x + sigma * y, -5, 5)
                    f_i = func(x_i)
                    total_calls += 1
                    pops.append(x_i)
                    fits.append(f_i)

                if len(fits) == 0:
                    break

                # Sort
                sorted_idx = np.argsort(fits)
                pops = [pops[i] for i in sorted_idx]
                fits = [fits[i] for i in sorted_idx]

                # Update best
                if fits[0] < best_fx:
                    best_fx = fits[0]
                    best_x = pops[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                generation += 1

                # Check for Stagnation and Degeneration
                cond_C = np.linalg.cond(C)
                
                # Trigger Local Search if stagnant
                if no_improve >= stagnation_threshold and total_calls + local_search_budget < budget:
                    if time.time() < deadline - 15:
                        local_x, local_f, used_evals = nelder_mead(best_x.copy(), local_search_budget, sigma)
                        total_calls += used_evals

                        if local_f < best_fx:
                            best_fx = local_f
                            best_x = local_x
                            no_improve = 0
                            x = best_x.copy()
                        else:
                            if no_improve > stagnation_threshold * 1.5:
                                restart_needed = True
                                break

                # Check for degeneration
                if sigma < 5e-5 or (cond_C > 5e5 and no_improve > 4):
                    if no_improve > 8:
                        restart_needed = True
                        break

                if total_calls >= budget:
                    break

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pops[i]

                # Update covariance and step size
                if sigma > 1e-10:
                    y_mean = (x_new - x) / sigma
                else:
                    y_mean = np.zeros(n)

                pc = (1 - c_c) * pc + c_c * y_mean
                ps = np.sqrt(1 - (1 - c_c)**2 * (1 - c_sigma)) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean

                norm_ps = np.linalg.norm(ps)
                hsig = (norm_ps / chi_n) < (1.4 + 2 / (n + 1))

                # Update C
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    d = pops[i] - x
                    C += c_mu * weights[i] * np.outer(d, d)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = max(1e-6, min(sigma, 10.0))

                x = x_new

            # If we broke out of the inner loop due to restart_needed and we have budget left
            if restart_needed and total_calls < budget:
                restart_count += 1
                continue
            else:
                break

        # Final high-precision polish if we have budget remaining
        if best_x is not None and total_calls < budget - 10:
            remaining = budget - total_calls
            polished_x, polished_f, used = final_polish(best_x.copy(), remaining)
            total_calls += used
            if polished_f < best_fx:
                best_fx = polished_f
                best_x = polished_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
