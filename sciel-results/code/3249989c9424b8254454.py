# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Coarse global exploration ---
        # Spend ~30 calls on global random sampling to find diverse starting points.
        n_global = 30
        candidates = []
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            used_calls += 1
            candidates.append((f, x))
            try_update(x, f)

        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # --- Phase 2: Multi-start Nelder-Mead ---
        # Select top K distinct points to start local searches
        K = 5
        starts = []
        for f, x in candidates:
            # Check if this point is sufficiently different from existing starts
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < 0.1:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((f, x))
            if len(starts) >= K:
                break

        if not starts:
            starts = [(candidates[0][0], candidates[0][1])]

        remaining_calls = budget - used_calls
        calls_per_start = max(10, remaining_calls // len(starts))

        def nelder_mead(x0, f0, max_calls):
            nonlocal used_calls, best_f, best_x
            
            # Initialize simplex
            alpha = 1.0
            simplex = [x0.copy()]
            f_vals = [f0]
            
            # Add other vertices
            for i in range(d):
                xi = x0.copy()
                xi[i] += alpha
                xi = np.clip(xi, -5, 5)
                if used_calls >= max_calls or check_time():
                    return
                fi = func(xi)
                used_calls += 1
                simplex.append(xi)
                f_vals.append(fi)
                try_update(xi, fi)
            
            # Nelder-Mead loop
            alpha = 1.0 # Expansion
            gamma = 2.0 # Contraction
            rho = -0.5  # Reflection
            sigma = 0.5 # Shrinkage
            
            while used_calls < max_calls and not check_time():
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if np.std(f_vals) < 1e-6:
                    break
                if np.linalg.norm(simplex[0] - simplex[1]) < 1e-6:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + rho * (simplex[-1] - centroid)
                xr = np.clip(xr, -5, 5)
                if used_calls >= max_calls or check_time():
                    break
                fr = func(xr)
                used_calls += 1
                try_update(xr, fr)
                
                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + alpha * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    if used_calls >= max_calls or check_time():
                        break
                    fe = func(xe)
                    used_calls += 1
                    try_update(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Reflection is good
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + gamma * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if used_calls >= max_calls or check_time():
                        break
                    fc = func(xc)
                    used_calls += 1
                    try_update(xc, fc)
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if used_calls >= max_calls or check_time():
                                return
                            fi = func(simplex[i])
                            used_calls += 1
                            f_vals[i] = fi
                            try_update(simplex[i], fi)
            
            # Update global best from this run
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = simplex[0].copy()

        # Run Nelder-Mead from each start
        for i, (f_start, x_start) in enumerate(starts):
            if check_time() or used_calls >= budget:
                break
            # Allocate calls, leaving some for final polish
            allocated = min(calls_per_start, budget - used_calls - 10)
            if allocated < 10:
                allocated = max(5, budget - used_calls)
            nelder_mead(x_start, f_start, used_calls + allocated)

        # --- Phase 3: Local polish with (1+1)-ES ---
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(50):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-7, 5.0))
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
