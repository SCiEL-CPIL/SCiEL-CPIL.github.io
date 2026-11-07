import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        d = self.dim
        N = self.budget

        call_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= N or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Bent Cigar has a narrow valley; random sampling helps locate it
        rng = np.random.default_rng(42)
        n_samples = 20
        for i in range(n_samples):
            if time.time() > deadline:
                break
            if call_count >= N:
                break
            x = rng.uniform(-5, 5, d)
            fx = wrapped_func(x)

        if call_count == 0:
            # Fallback if something went wrong immediately
            wrapped_func(np.zeros(d))

        # Phase 2: Local optimization with L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(d)]
        remaining_budget = N - call_count

        try:
            if remaining_budget > 10:
                # L-BFGS-B is efficient for smooth bounded problems
                # Use a generous maxiter but let the wrapper enforce the budget
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfev': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
        except Exception:
            pass

        # Phase 3: Coordinate Descent with Golden Section Search
        # This exploits the anisotropy of Bent Cigar.
        # Bent Cigar is roughly y1^2 + sum(1000*yi^2).
        # The valley is extremely narrow in dimensions 2..d.
        # Coordinate descent decouples dimensions.
        # We use Golden Section Search for 1D minimization on each axis.
        
        if call_count < N and time.time() < deadline:
            x_current = best_x.copy()
            f_current = best_f
            
            # Iterate through dimensions. We might need multiple passes.
            # Let's do a few passes until budget is low or no improvement.
            max_cd_passes = 10
            for p in range(max_cd_passes):
                if call_count >= N or time.time() > deadline:
                    break
                
                improved = False
                for i in range(d):
                    if call_count >= N or time.time() > deadline:
                        break
                    
                    # Define 1D function
                    def f1d(t, x_base=x_current, axis=i):
                        x_test = x_base.copy()
                        x_test[axis] = t
                        return wrapped_func(x_test)
                    
                    # Golden section search
                    # Initial bracket: [x_current[i] - 1.0, x_current[i] + 1.0]
                    # Or maybe smaller if we are close?
                    # Let's use a moderate initial step.
                    # If we are far, L-BFGS-B should have brought us close.
                    # If we are very close, a smaller step is better.
                    # Let's start with step 0.5.
                    a = x_current[i] - 0.5
                    b = x_current[i] + 0.5
                    
                    # Ensure bounds
                    a = max(a, -5.0)
                    b = min(b, 5.0)
                    
                    if b - a < 1e-6:
                        continue

                    # Golden section search parameters
                    gr = (np.sqrt(5) - 1) / 2  # 0.618
                    c = b - gr * (b - a)
                    dd = a + gr * (b - a)
                    
                    fc = f1d(c)
                    fd = f1d(dd)
                    
                    # We want to minimize
                    # If fc < fd, the minimum is in [a, dd]
                    # If fd < fc, the minimum is in [c, b]
                    
                    # Iterate until interval is small or budget runs out
                    max_iter_1d = 20
                    for _ in range(max_iter_1d):
                        if call_count >= N or time.time() > deadline:
                            break
                        if b - a < 1e-6:
                            break
                        if fc < fd:
                            b = dd
                            dd = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = f1d(c)
                        else:
                            a = c
                            c = dd
                            fc = fd
                            dd = a + gr * (b - a)
                            fd = f1d(dd)
                    
                    # Update x_current with the best point found in 1D search
                    # The best point is the one with lower f among c and dd
                    if fc < fd:
                        x_new_val = c
                        f_new_val = fc
                    else:
                        x_new_val = dd
                        f_new_val = fd
                    
                    # Only update if we actually improved
                    if f_new_val < f_current:
                        x_current[i] = x_new_val
                        f_current = f_new_val
                        improved = True
                        # Update global best
                        if f_current < best_f:
                            best_f = f_current
                            best_x = x_current.copy()
                    
                if not improved:
                    break # No improvement in this pass, stop

        # Phase 4: Final Nelder-Mead polish if budget remains
        remaining_budget = N - call_count
        if remaining_budget > 5 and time.time() < deadline:
            try:
                # Use a small initial simplex for local polish
                x0 = best_x.copy()
                # Create a small simplex around the current best
                step = 0.01
                simplex = [x0.copy()]
                for i in range(d):
                    xi = x0.copy()
                    xi[i] += step
                    simplex.append(xi)
                
                # scipy minimize with Nelder-Mead
                # We pass x0, but it constructs its own simplex.
                # To use our custom simplex, we might need to implement NM ourselves
                # or just let scipy handle it. SciPy's NM is robust.
                result2 = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 50), 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
