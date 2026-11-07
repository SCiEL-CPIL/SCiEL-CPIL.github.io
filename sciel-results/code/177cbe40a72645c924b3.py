"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global random sampling to find good basins
        # Use 15 diverse initial points
        n_initial = min(15, budget // 20)
        candidates = []
        for i in range(n_initial):
            if eval_count >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            # Fallback: single random point
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local refinement on top candidates
        # Allocate remaining budget across top 3 candidates
        top_k = min(3, len(candidates))
        remaining = budget - eval_count
        per_candidate = max(20, remaining // max(1, top_k))

        for rank in range(top_k):
            if eval_count >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            # Nelder-Mead simplex
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Build simplex
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                delta = 0.5 if x0[j] + 0.5 <= 5 else -0.5
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            # Evaluate simplex
            fvals = [f0]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            # Nelder-Mead iterations
            max_nm_iter = per_candidate - (len(fvals) - 1)
            for _ in range(max(0, max_nm_iter)):
                if eval_count >= budget or not time_ok():
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if eval_count >= budget or not time_ok():
                    break
                fr = evaluate(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            if eval_count >= budget or not time_ok():
                                break
                        if eval_count >= budget or not time_ok():
                            break
                        fvals = [f0]
                        for pt in simplex[1:]:
                            fx = evaluate(pt)
                            if fx is None:
                                break
                            fvals.append(fx)
                        if len(fvals) != len(simplex):
                            break

        # Phase 3: If budget remains, do a final local polish around best
        if best_x is not None and eval_count < budget and time_ok():
            # Small step refinement with adaptive step size
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            for _ in range(budget - eval_count):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
