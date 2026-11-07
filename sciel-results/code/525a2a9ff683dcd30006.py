import numpy as np
import time
import scipy.optimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        bounds = [(-5, 5)] * dim

        # Phase 1: Global Exploration
        # Use a mix of random points and a coarse grid to cover the space
        # The minimizer is in [-4, 4]^5, so we focus there but allow edges
        n_random = 60
        points = []
        
        # Random points
        for _ in range(n_random):
            points.append(rng.uniform(-4.5, 4.5, dim))
            
        # Deterministic structured points (Halton-like or just grid corners/centers)
        # In 5D, a full grid is expensive. Let's use a few key structural points.
        # Center
        points.append(np.zeros(dim))
        # Corners of [-4,4]^5 are 32 points. Let's add a subset or all if budget allows.
        # 32 corners + 60 random = 92 points. This is fine.
        corners = []
        for i in range(2**dim):
            corner = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            corner = corner * 8 - 4  # Map {0,1} to {-4, 4}
            points.append(corner)
        
        # Evaluate all points
        for x0 in points:
            if call_count >= self.budget - 50:
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            tracked_func(best_x)

        # Phase 2: Local Optimization
        # Use Nelder-Mead as the primary local optimizer as per directive, 
        # but also try L-BFGS-B if budget remains, or vice versa.
        # Nelder-Mead is derivative-free and can handle the rotated landscape well 
        # if initialized close enough. L-BFGS-B uses gradients which might be noisy 
        # if the function is evaluated with some noise, but here it's deterministic.
        # However, the directive specifically suggests Nelder-Mead + sparse global lattice.
        
        # Let's run Nelder-Mead from the best point found.
        if time.time() < deadline and call_count < self.budget - 20:
            remaining = self.budget - call_count - 10
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'maxfev': remaining,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: If budget remains, try L-BFGS-B from the new best point
        # L-BFGS-B often converges faster in the final stages for smooth functions.
        if time.time() < deadline and call_count < self.budget - 10:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-15,
                            'eps': 1e-8
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        # Phase 4: Final polish with Powell if any budget left
        if time.time() < deadline and call_count < self.budget - 5:
            remaining = self.budget - call_count - 2
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='Powell',
                        bounds=bounds,
                        options={
                            'maxfev': remaining,
                            'xtol': 1e-12,
                            'ftol': 1e-12
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
