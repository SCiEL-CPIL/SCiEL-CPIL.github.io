import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if not np.isfinite(f):
                return f
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def nelder_mead(x0, max_calls=40, xatol=1e-6, fatol=1e-8):
            """Hand-rolled Nelder-Mead to avoid scipy overhead."""
            if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                return x0.copy()
            
            n = dim
            alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
            base_step = 0.5
            
            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += base_step
            
            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fvals[i] = wrapped_func(simplex[i])
                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                    return self.best_x.copy()
            
            # Main loop
            for iteration in range(max_calls * (n + 1)):
                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                    break
                
                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Convergence check
                if np.all(np.abs(fvals - fvals[0]) < fatol):
                    break
                if np.all(np.abs(simplex - simplex[0]) < xatol):
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                fr = wrapped_func(xr)
                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    fe = wrapped_func(xe)
                    if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                        break
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    if fr < fvals[n]:
                        # Outside contraction
                        xc = centroid + sigma * (xr - centroid)
                        fc = wrapped_func(xc)
                        if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                            break
                        if fc <= fr:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = wrapped_func(simplex[i])
                                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                                    break
                    else:
                        # Inside contraction
                        xc = centroid - sigma * (centroid - simplex[n])
                        fc = wrapped_func(xc)
                        if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                            break
                        if fc <= fvals[n]:
                            simplex[n] = xc
                            fvals[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = wrapped_func(simplex[i])
                                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                                    break
            
            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy()

        def coordinate_descent(x, max_calls=30):
            """Improved coordinate descent with golden-section search."""
            if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                return x.copy()
            
            x = x.copy()
            f = wrapped_func(x)
            if not np.isfinite(f):
                return x
            
            for coord in range(dim):
                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                    break
                
                # Golden-section search along this coordinate
                lo = x[coord] - 1.0
                hi = x[coord] + 1.0
                gr = (np.sqrt(5) - 1) / 2  # ~0.618
                
                for _ in range(8):
                    if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                        break
                    
                    m1 = lo + gr * (hi - lo)
                    m2 = hi - gr * (hi - lo)
                    
                    x1 = x.copy()
                    x1[coord] = m1
                    f1 = wrapped_func(x1)
                    
                    x2 = x.copy()
                    x2[coord] = m2
                    f2 = wrapped_func(x2)
                    
                    if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                        break
                    
                    if f1 < f2:
                        hi = m2
                    else:
                        lo = m1
                
                # Evaluate midpoint
                x_mid = x.copy()
                x_mid[coord] = (lo + hi) / 2
                f_mid = wrapped_func(x_mid)
                if np.isfinite(f_mid) and f_mid < f:
                    x = x_mid
                    f = f_mid
            
            return x

        # Generate diverse starting points
        np.random.seed(42)
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(4):
            starts.append(np.random.uniform(-4, 4, dim))
        starts.append(np.random.uniform(-5, -3, dim))
        starts.append(np.random.uniform(3, 5, dim))

        num_starts = len(starts)

        for i in range(num_starts):
            if self.calls >= budget - 15:
                break
            if time.time() > self.deadline - 15:
                break

            x0 = starts[i].copy()
            f0 = wrapped_func(x0)

            if not np.isfinite(f0):
                continue

            # Phase 1: Nelder-Mead local optimization
            if self.calls < budget - 25 and time.time() < self.deadline - 15:
                try:
                    remaining = min(45, budget - self.calls - 15)
                    if remaining > 20:
                        nm_calls = min(remaining, 40)
                        result = nelder_mead(x0, max_calls=nm_calls, xatol=1e-6, fatol=1e-8)
                        if np.isfinite(self.best_f):
                            self.best_x = np.clip(result, -5, 5).copy()
                except Exception:
                    pass

            # Phase 2: Coordinate descent for fine-tuning
            if self.calls < budget - 15 and time.time() < self.deadline - 10:
                try:
                    x_cd = coordinate_descent(self.best_x, max_calls=25)
                    f_cd = wrapped_func(x_cd)
                    if np.isfinite(f_cd) and f_cd < self.best_f:
                        self.best_f = f_cd
                        self.best_x = np.clip(x_cd, -5, 5).copy()
                except Exception:
                    pass

        # Final high-precision polish
        if self.calls < budget - 5 and time.time() < self.deadline - 5:
            try:
                # Tight NM
                remaining = min(25, budget - self.calls - 3)
                if remaining > 10:
                    result = nelder_mead(self.best_x, max_calls=remaining, xatol=1e-8, fatol=1e-10)
                    f_final = wrapped_func(result)
                    if np.isfinite(f_final) and f_final < self.best_f:
                        self.best_f = f_final
                        self.best_x = np.clip(result, -5, 5).copy()
                
                # Final coordinate descent
                if self.calls < budget - 3 and time.time() < self.deadline - 3:
                    x_cd = coordinate_descent(self.best_x, max_calls=15)
                    f_cd = wrapped_func(x_cd)
                    if np.isfinite(f_cd) and f_cd < self.best_f:
                        self.best_f = f_cd
                        self.best_x = np.clip(x_cd, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
