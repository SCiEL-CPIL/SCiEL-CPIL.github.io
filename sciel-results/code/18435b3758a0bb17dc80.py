"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step sizing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        # We track all function calls to stay within budget
        call_count = [0]

        def safe_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        # Phase 1: Initial sampling to find a good basin
        # Use a small set of diverse starting points
        rng = np.random.default_rng(42)

        # Start with a few candidate points
        candidates = []
        # Center
        candidates.append(np.zeros(dim))
        # A few random points
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))

        # Evaluate all candidates
        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = safe_call(c)
            if fc is None:
                break
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            # Fallback if we ran out of budget immediately
            return np.zeros(dim), np.inf

        # Phase 2: Nelder-Mead local refinement from the best point
        # Build initial simplex
        simplex = [best_x.copy()]
        base = best_x
        for i in range(dim):
            point = base.copy()
            if point[i] + 0.5 <= 5.0:
                point[i] += 0.5
            else:
                point[i] -= 0.5
            simplex.append(np.clip(point, -5, 5))

        # Evaluate simplex
        simplex_f = []
        for i in range(len(simplex)):
            if simplex_f:
                fi = safe_call(simplex[i])
            else:
                # First point is best_x, we already have its value
                fi = best_f
            if fi is None:
                # Ran out of budget
                return best_x, best_f
            simplex_f.append(fi)

        # Nelder-Mead iterations
        # We'll use a simplified version that doesn't require scipy
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        max_nm_iterations = 200

        for _ in range(max_nm_iterations):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Check convergence
            if len(simplex_f) > 1:
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-10:
                    break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break

            # Expansion
            if fr < simplex_f[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            # Contraction
            elif fr < simplex_f[-2]:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= fr:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if call_count[0] >= budget:
                            break
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        simplex_f[i] = fi
                    break
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    if call_count[0] >= budget:
                        break
                    fi = safe_call(simplex[i])
                    if fi is None:
                        break
                    simplex_f[i] = fi
                break

            # Track best
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

        # Phase 3: If we have remaining budget, do a few more restarts
        # from near the best point with smaller perturbations
        remaining = budget - call_count[0]
        if remaining > 10:
            for _ in range(min(5, remaining // 10)):
                if time.time() > deadline:
                    break
                # Small perturbation around best_x
                perturbation = 0.1
                x_new = best_x + perturbation * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                fx_new = safe_call(x_new)
                if fx_new is None:
                    break
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()

                    # Quick Nelder-Mead from this new point
                    simplex2 = [best_x.copy()]
                    for i in range(dim):
                        point = best_x.copy()
                        if point[i] + 0.1 <= 5.0:
                            point[i] += 0.1
                        else:
                            point[i] -= 0.1
                        simplex2.append(np.clip(point, -5, 5))

                    simplex2_f = []
                    for i in range(len(simplex2)):
                        if simplex2_f:
                            fi = safe_call(simplex2[i])
                        else:
                            fi = best_f
                        if fi is None:
                            break
                        simplex2_f.append(fi)

                    for _ in range(50):
                        if call_count[0] >= budget:
                            break
                        if time.time() > deadline:
                            break

                        order = np.argsort(simplex2_f)
                        simplex2 = [simplex2[i] for i in order]
                        simplex2_f = [simplex2_f[i] for i in order]

                        if len(simplex2_f) > 1:
                            f_range = simplex2_f[-1] - simplex2_f[0]
                            if f_range < 1e-10:
                                break

                        centroid2 = np.mean(simplex2[:-1], axis=0)

                        xr2 = centroid2 + alpha * (centroid2 - simplex2[-1])
                        xr2 = np.clip(xr2, -5, 5)
                        fr2 = safe_call(xr2)
                        if fr2 is None:
                            break

                        if fr2 < simplex2_f[0]:
                            xe2 = centroid2 + gamma * (centroid2 - simplex2[-1])
                            xe2 = np.clip(xe2, -5, 5)
                            fe2 = safe_call(xe2)
                            if fe2 is None:
                                break
                            if fe2 < fr2:
                                simplex2[-1] = xe2
                                simplex2_f[-1] = fe2
                            else:
                                simplex2[-1] = xr2
                                simplex2_f[-1] = fr2
                        elif fr2 < simplex2_f[-2]:
                            xc2 = centroid2 + rho * (simplex2[-1] - centroid2)
                            xc2 = np.clip(xc2, -5, 5)
                            fc2 = safe_call(xc2)
                            if fc2 is None:
                                break
                            if fc2 <= fr2:
                                simplex2[-1] = xc2
                                simplex2_f[-1] = fc2
                            else:
                                for i in range(1, len(simplex2)):
                                    simplex2[i] = simplex2[0] + sigma * (simplex2[i] - simplex2[0])
                                    simplex2[i] = np.clip(simplex2[i], -5, 5)
                                    if call_count[0] >= budget:
                                        break
                                    fi2 = safe_call(simplex2[i])
                                    if fi2 is None:
                                        break
                                    simplex2_f[i] = fi2
                                break
                        else:
                            for i in range(1, len(simplex2)):
                                simplex2[i] = simplex2[0] + sigma * (simplex2[i] - simplex2[0])
                                simplex2[i] = np.clip(simplex2[i], -5, 5)
                                if call_count[0] >= budget:
                                    break
                                fi2 = safe_call(simplex2[i])
                                if fi2 is None:
                                    break
                                simplex2_f[i] = fi2
                            break

                        min_idx2 = np.argmin(simplex2_f)
                        if simplex2_f[min_idx2] < best_f:
                            best_f = simplex2_f[min_idx2]
                            best_x = simplex2[min_idx2].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
