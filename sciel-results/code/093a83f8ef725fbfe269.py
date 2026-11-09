"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage global-to-local search: Sobol sampling then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        # Helper to safely call func
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            val = func(x)
            calls_made += 1
            return val

        # Stage 1: Sobol quasi-random sampling
        # Reduced to 150 to allow more room for local optimization
        n_samples = min(150, budget // 3)
        # Generate Sobol sequence
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_samples = sampler.random(n_samples)
        except ImportError:
            # Fallback to random if scipy not available
            sobol_samples = rng.random((n_samples, dim))

        # Map to [-5, 5]
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Evaluate all samples
        values = np.empty(n_samples)
        for i in range(n_samples):
            if calls_made >= budget or time.time() > deadline - 60:
                # Pad with infinity if we run out of budget early
                values[i:] = np.inf
                break
            x = sobol_samples[i]
            val = safe_call(x)
            if val is None:
                values[i:] = np.inf
                break
            values[i] = val

        # Get top 5 candidates
        valid_indices = np.where(np.isfinite(values))[0]
        if len(valid_indices) == 0:
            return sobol_samples[0], np.inf

        n_top = min(5, len(valid_indices))
        # Sort only valid ones
        sorted_valid_indices = valid_indices[np.argsort(values[valid_indices])][:n_top]
        top_points = sobol_samples[sorted_valid_indices]
        top_values = values[sorted_valid_indices]

        # Stage 2: Nelder-Mead local optimization from top candidates
        best_x = top_points[0]
        best_f = top_values[0]

        # Distribute remaining budget among top candidates
        remaining_budget = budget - calls_made
        calls_per_candidate = max(10, remaining_budget // n_top)

        for k in range(n_top):
            if time.time() > deadline - 30 or calls_made >= budget:
                break

            x0 = top_points[k].copy()
            f0 = top_values[k]

            # Adaptive initial simplex size
            if k == 0:
                simplex_size = 1.0
            else:
                # Scale by distance to best point
                dist = np.linalg.norm(x0 - best_x)
                simplex_size = max(0.5, min(3.0, dist * 0.5))

            # Implement Nelder-Mead manually to control budget
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]

            for i in range(dim):
                if calls_made >= budget:
                    break
                perturbation = x0.copy()
                if abs(perturbation[i]) > 0.01:
                    perturbation[i] *= 1.05
                else:
                    perturbation[i] += 0.05 * simplex_size
                simplex.append(perturbation)
                val = safe_call(perturbation)
                if val is None:
                    # Budget exhausted, remove this point from simplex
                    simplex.pop()
                    break
                simplex_f.append(val)

            # If we didn't build a full simplex, skip NM
            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_nm_iters = calls_per_candidate

            for _ in range(max_nm_iters):
                if time.time() > deadline - 20 or calls_made >= budget:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        # Budget ran out during expansion, keep reflection
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            break
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            shrink_ok = True
                            for i in range(1, dim + 1):
                                if calls_made >= budget:
                                    shrink_ok = False
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                val = safe_call(simplex[i])
                                if val is None:
                                    shrink_ok = False
                                    break
                                simplex_f[i] = val
                            if not shrink_ok:
                                break

                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

        # Final check
        if time.time() > deadline:
            return best_x, best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
