import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with Dual Annealing
        de_budget = int(budget * 0.5)  # 50% of budget for global search
        if de_budget > 20:
            try:
                bounds = [(-4.0, 4.0)] * dim
                da_result = dual_annealing(
                    safe_func,
                    bounds,
                    maxiter=50,
                    maxfev=de_budget,
                    seed=42,
                    restarts=3
                )
            except Exception:
                pass

        if not np.isfinite(best_f):
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)

        if not np.isfinite(best_f):
            return best_x, best_f

        # Phase 2: Local refinement with L-BFGS-B
        local_budget = budget - calls
        if local_budget > 10:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': local_budget,
                        'gtol': 1e-12,
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 3: Adaptive line search along Halton directions for final polish
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline:
            # Generate Halton sequence directions
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            # Create Halton directions in [0,1]^dim, then map to [-1,1]^dim
            n_halton = min(remaining // 10, 30)  # Limit to reasonable number of directions
            halton_dirs = []
            for i in range(1, n_halton + 1):
                d = np.array([halton(i, p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
                d = 2.0 * d - 1.0  # Map to [-1, 1]
                norm = np.linalg.norm(d)
                if norm > 1e-10:
                    d /= norm
                halton_dirs.append(d)

            # For each direction, do a golden-section search
            phi = (1 + np.sqrt(5)) / 2
            inv_phi = 1.0 / phi
            inv_phi2 = 1.0 / (phi * phi)

            for direction in halton_dirs:
                if remaining < 5 or time.time() > deadline:
                    break

                # Golden section search along the direction from best_x
                # Search range: [-5, 5] along the direction
                a = -5.0
                b = 5.0
                c = b - inv_phi * (b - a)
                d = a + inv_phi * (b - a)

                # Evaluate at c and d
                xc = best_x + c * direction
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_func(xc)

                xd = best_x + d * direction
                xd = np.clip(xd, -5.0, 5.0)
                fd = safe_func(xd)

                # Iterate golden section search
                max_iter = min(remaining - 2, 20)
                for _ in range(max_iter):
                    if remaining < 3 or time.time() > deadline:
                        break

                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - inv_phi * (b - a)
                        xc = best_x + c * direction
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_func(xc)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + inv_phi * (b - a)
                        xd = best_x + d * direction
                        xd = np.clip(xd, -5.0, 5.0)
                        fd = safe_func(xd)

                # After golden section, the best point is at min(fc, fd)
                # Update best_x if we found something better
                # safe_func already tracks best_x and best_f

        # Phase 4: Final high-precision polish with BFGS
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='BFGS',
                    options={
                        'maxiter': 100,
                        'maxfun': remaining,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        return best_x, best_f
