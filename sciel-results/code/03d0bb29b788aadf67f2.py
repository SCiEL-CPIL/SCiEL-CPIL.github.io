"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-start, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = dim

        # Phase 1: Global random sampling to find good basins
        n_initial = min(20, budget // 15)
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = safe_call(x)
            if fx is not None:
                candidates.append((fx, x.copy()))
        
        # Also try some structured points
        structured = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-2, 2, n),
        ]
        for x_struct in structured:
            if calls >= budget or not time_ok():
                break
            x = np.clip(x_struct, -5, 5)
            fx = safe_call(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            safe_call(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local refinement on top candidates
        top_k = min(4, len(candidates))
        remaining = budget - calls
        per_candidate = max(40, remaining // max(1, top_k + 1))

        for rank in range(top_k):
            if calls >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build simplex with random perturbations for better coverage
            simplex = [x0.copy()]
            base_delta = 0.3
            for j in range(n):
                # Use random direction with adaptive magnitude
                direction = rng.choice([-1, 1])
                delta = direction * base_delta * rng.uniform(0.5, 1.5)
                
                # Adjust for boundary
                dist_to_pos = 5.0 - x0[j]
                dist_to_neg = x0[j] + 5.0
                if delta > 0 and delta > dist_to_pos:
                    delta = dist_to_pos * 0.8
                elif delta < 0 and -delta > dist_to_neg:
                    delta = -dist_to_neg * 0.8
                    
                point = x0.copy()
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if calls >= budget or not time_ok():
                    break
                fx = safe_call(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            max_nm_iter = per_candidate - (len(fvals) - 1)
            prev_best = f0
            stagnant = 0
            
            for _ in range(max(0, max_nm_iter)):
                if calls >= budget or not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if calls >= budget or not time_ok():
                    break
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Check for improvement
                    if fr < prev_best - 1e-10:
                        prev_best = fr
                        stagnant = 0
                    else:
                        stagnant += 1
                        
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if calls >= budget or not time_ok():
                        break
                    fe = safe_call(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                        if fe < prev_best - 1e-10:
                            prev_best = fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                    if fr < prev_best - 1e-10:
                        prev_best = fr
                        stagnant = 0
                    else:
                        stagnant += 1
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if calls >= budget or not time_ok():
                        break
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                        if fc < prev_best - 1e-10:
                            prev_best = fc
                            stagnant = 0
                        else:
                            stagnant += 1
                    else:
                        # Shrink simplex
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                            if calls >= budget or not time_ok():
                                break
                        if calls >= budget or not time_ok():
                            break
                        fvals_new = []
                        for pt in simplex:
                            if pt is simplex[0]:
                                fvals_new.append(fvals[0])
                            else:
                                fx = safe_call(pt)
                                if fx is None:
                                    break
                                fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new
                        stagnant += 1

                # Early stop if stagnant for many iterations
                if stagnant > 15:
                    break

        # Phase 3: Final (1+1)-ES polish on the best point found
        if calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.05
            min_sigma = 1e-10
            max_sigma = 0.5
            success_count = 0
            total_count = 0
            
            for _ in range(budget - calls):
                if not time_ok():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                total_count += 1
                if fy < f:
                    x, f = y, fy
                    success_count += 1
                    # Increase sigma on success, but cap it
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.85, min_sigma)
                
                # Adaptive adjustment based on success rate
                if total_count >= 10:
                    success_rate = success_count / total_count
                    if success_rate > 0.7:
                        sigma = min(sigma * 1.05, max_sigma)
                    elif success_rate < 0.2:
                        sigma = max(sigma * 0.95, min_sigma)
                    
                    # Reset counters periodically
                    if total_count >= 20:
                        success_count = 0
                        total_count = 0
                
                if sigma < min_sigma * 1.1:
                    break
        
        # Phase 4: Coordinate descent polish for high precision
        if calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in step_sizes:
                if calls >= budget or not time_ok():
                    break
                
                improved = True
                while improved and calls < budget and time_ok():
                    improved = False
                    for j in range(n):
                        if calls >= budget or not time_ok():
                            break
                        
                        # Try positive step
                        x_try = x.copy()
                        x_try[j] = min(x[j] + step, 5.0)
                        fy = safe_call(x_try)
                        if fy is None:
                            break
                        if fy < f - 1e-12:
                            x, f = x_try, fy
                            improved = True
                            continue
                        
                        # Try negative step
                        x_try = x.copy()
                        x_try[j] = max(x[j] - step, -5.0)
                        fy = safe_call(x_try)
                        if fy is None:
                            break
                        if fy < f - 1e-12:
                            x, f = x_try, fy
                            improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END
