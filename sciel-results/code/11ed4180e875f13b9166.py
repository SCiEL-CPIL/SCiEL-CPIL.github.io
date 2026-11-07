"""
Griewank-Rosenbrock aware hybrid optimizer.
Uses structured global sampling, Nelder-Mead with adaptive restarts,
and gradient-based polish for the smooth but multimodal landscape.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s

        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def budget_ok():
            return calls < budget

        # --- Phase 1: Structured global sampling ---
        # Use a combination of grid points, LHS, and structured points
        n_lhs = min(15, max(8, budget // 20))
        
        # LHS points
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                lo = i / n_lhs
                hi = (i + 1) / n_lhs
                point[d] = rng.uniform(lo, hi)
            point = point * 10 - 5
            lhs_points.append(point)
        
        # Structured points: center, corners, and some intermediate
        structured = [
            np.zeros(dim),
            np.full(dim, -4.0),
            np.full(dim, 4.0),
            np.full(dim, -2.0),
            np.full(dim, 2.0),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
            np.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
        ]
        
        all_initial = lhs_points + structured
        init_results = []
        for p in all_initial:
            if not budget_ok() or not time_ok():
                break
            fx = safe_call(p)
            init_results.append((fx, p.copy()))
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Sort by fitness, take top candidates for local search
        init_results.sort(key=lambda x: x[0])
        top_k = min(5, len(init_results))
        candidates = [p for _, p in init_results[:top_k]]
        
        # Add a few random perturbations of the best
        for _ in range(3):
            if not budget_ok() or not time_ok():
                break
            x_pert = best_x + 0.3 * rng.standard_normal(dim)
            x_pert = np.clip(x_pert, -5, 5)
            candidates.append(x_pert)

        # --- Phase 2: Nelder-Mead local search from each candidate ---
        def nelder_mead(x0, max_iter=80, tol=1e-10):
            nonlocal best_f, best_x, calls
            if not budget_ok() or not time_ok():
                return x0.copy(), best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            for d in range(dim):
                x_new = x0.copy()
                step = 0.1 if x0[d] >= 0 else -0.1
                x_new[d] += step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
            
            # Evaluate initial simplex
            for i, v in enumerate(simplex):
                if not budget_ok() or not time_ok():
                    break
                simplex[i] = (safe_call(v), v)
            
            simplex.sort(key=lambda x: x[0])
            f0, x0_curr = simplex[0]
            if f0 < best_f:
                best_f = f0
                best_x = x0_curr.copy()
            
            for it in range(max_iter):
                if not budget_ok() or not time_ok():
                    break
                
                # Check convergence
                f_best = simplex[0][0]
                f_worst = simplex[-1][0]
                if abs(f_worst - f_best) < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean([v for _, v in simplex[:-1]], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = np.clip(centroid + alpha * (centroid - simplex[-1][1]), -5, 5)
                if not budget_ok() or not time_ok():
                    break
                f_ref = safe_call(x_ref)
                
                if f_best <= f_ref < simplex[-2][0]:
                    simplex[-1] = (f_ref, x_ref)
                elif f_ref < f_best:
                    # Expansion
                    gamma = 2.0
                    x_exp = np.clip(centroid + gamma * (centroid - simplex[-1][1]), -5, 5)
                    if not budget_ok() or not time_ok():
                        break
                    f_exp = safe_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = (f_exp, x_exp)
                    else:
                        simplex[-1] = (f_ref, x_ref)
                else:
                    # Contraction
                    beta = 0.5
                    if f_ref < simplex[-1][0]:
                        x_con = np.clip(centroid + beta * (x_ref - centroid), -5, 5)
                    else:
                        x_con = np.clip(centroid - beta * (simplex[-1][1] - centroid), -5, 5)
                    if not budget_ok() or not time_ok():
                        break
                    f_con = safe_call(x_con)
                    if f_con < min(f_ref, simplex[-1][0]):
                        simplex[-1] = (f_con, x_con)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i][1] = np.clip(simplex[0][1] + 0.5 * (simplex[i][1] - simplex[0][1]), -5, 5)
                            if not budget_ok() or not time_ok():
                                break
                            simplex[i][0] = safe_call(simplex[i][1])
                
                simplex.sort(key=lambda x: x[0])
                f0, x0_curr = simplex[0]
                if f0 < best_f:
                    best_f = f0
                    best_x = x0_curr.copy()
            
            return x0_curr, f0

        # Run Nelder-Mead from each candidate
        for cand in candidates:
            if not budget_ok() or not time_ok():
                break
            nelder_mead(cand, max_iter=60)

        # --- Phase 3: Gradient-based polish ---
        def gradient_polish(x0, max_iter=40):
            nonlocal best_f, best_x, calls
            if not budget_ok() or not time_ok():
                return x0.copy(), best_f
            
            x = x0.copy()
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            h = 1e-5
            lr = 0.1
            for it in range(max_iter):
                if not budget_ok() or not time_ok():
                    break
                
                # Compute gradient
                grad = np.zeros(dim)
                for d in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[d] += h
                    xm[d] -= h
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    if not budget_ok() or not time_ok():
                        break
                    fp = safe_call(xp)
                    fm = safe_call(xm)
                    if np.isfinite(fp) and np.isfinite(fm):
                        grad[d] = (fp - fm) / (2 * h)
                
                if not budget_ok() or not time_ok():
                    break
                
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break
                
                # Gradient descent step
                x_new = np.clip(x - lr * grad / (gnorm + 1e-10), -5, 5)
                if not budget_ok() or not time_ok():
                    break
                fx_new = safe_call(x_new)
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    lr *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
            
            return x, fx

        # Polish from best and a few nearby points
        gradient_polish(best_x.copy(), max_iter=30)
        
        # Try polishing from a few random points near the best
        for _ in range(2):
            if not budget_ok() or not time_ok():
                break
            x_near = best_x + 0.1 * rng.standard_normal(dim)
            x_near = np.clip(x_near, -5, 5)
            gradient_polish(x_near, max_iter=20)

        # --- Phase 4: Final fine-tuning ---
        if best_x is not None and budget_ok() and time_ok():
            x = best_x.copy()
            fx = best_f
            local_sigma = 0.05
            max_local = min(20, budget - calls)
            
            for _ in range(max_local):
                if not time_ok() or not budget_ok():
                    break
                y = np.clip(x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    local_sigma *= 1.02
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    local_sigma *= 0.7
                    if local_sigma < 1e-6:
                        break

        return best_x, best_f
