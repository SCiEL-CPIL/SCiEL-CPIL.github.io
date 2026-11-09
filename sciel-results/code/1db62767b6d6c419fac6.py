import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B + Valley-Following Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0
        self._cache = {}

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0
        self._cache = {}

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            key = tuple(x.tolist())
            if key in self._cache:
                return self._cache[key]
            
            self.calls_made += 1
            val = func(x)
            self._cache[key] = val
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def valley_follow(x0, max_iters=20):
            """Budget-aware adaptive valley-following for Rosenbrock."""
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            step_size = 0.5
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Adaptive valley-following: blend toward x[i+1] = x[i]^2
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 5:
                        break
                    target_next = x[i] ** 2
                    blend = 0.7  # strong pull toward valley
                    x_vf = x.copy()
                    x_vf[i+1] = (1 - blend) * x[i+1] + blend * target_next
                    x_vf = np.clip(x_vf, -5, 5)
                    f_vf = wrapped_func(x_vf)
                    if f_vf < f:
                        x = x_vf
                        f = f_vf
                        improved = True
                        step_size = min(step_size * 1.2, 1.0)
                    else:
                        step_size = max(step_size * 0.7, 0.05)
                
                # Efficient coordinate descent with adaptive steps
                for i in range(self.dim):
                    if self.calls_made >= self.budget - 5:
                        break
                    best_local_f = f
                    best_local_x = x.copy()
                    # Try 3 step sizes in both directions (6 evals per coordinate)
                    for step_mult in [step_size, step_size * 0.5, step_size * 0.25]:
                        if self.calls_made >= self.budget - 5:
                            break
                        x_new = x.copy()
                        x_new[i] += step_mult
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped_func(x_new)
                        if f_new < best_local_f:
                            best_local_f = f_new
                            best_local_x = x_new
                        if self.calls_made >= self.budget - 5:
                            break
                        x_new = x.copy()
                        x_new[i] -= step_mult
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped_func(x_new)
                        if f_new < best_local_f:
                            best_local_f = f_new
                            best_local_x = x_new
                    if best_local_f < f:
                        x = best_local_x
                        f = best_local_f
                        improved = True
                
                if not improved:
                    step_size = max(step_size * 0.5, 0.01)
                    if step_size < 0.02:
                        break
            
            return x, f

        rng = np.random.default_rng(42)
        
        # Structured starts: valley points + perturbations + random
        starts = [np.ones(self.dim)]
        # Points along the Rosenbrock valley: x[0]=a, x[i+1]=a^2
        for a in [0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 0.3, -0.3, 1.5, -1.5]:
            v = np.zeros(self.dim)
            v[0] = a
            for i in range(1, self.dim):
                v[i] = a ** (2 ** i)
            v = np.clip(v, -5, 5)
            starts.append(v)
            starts.append(v + rng.normal(0, 0.3, self.dim))
        # Random starts
        for _ in range(4):
            starts.append(rng.uniform(-3, 5, self.dim))
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-4):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 12:
                continue
            
            # Phase 1: L-BFGS-B with tighter settings for faster convergence
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 60,
                        'maxls': 15,
                        'ftol': 1e-9,
                        'gtol': 1e-7,
                        'maxfun': 400
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                break
            except Exception:
                pass
            
            # Phase 2: Valley-following refinement
            if self.best_x is not None and self.calls_made < self.budget - 8:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=12)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # Final high-precision polish
        if self.best_x is not None and self.calls_made < self.budget - 5 and time.time() < self.deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxls': 20,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': 300
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                pass
            except Exception:
                pass
            
            # Final valley-following polish
            if self.best_x is not None and self.calls_made < self.budget - 3:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=8)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    pass
                except Exception:
                    pass
            
            # Extra L-BFGS-B polish if budget allows
            if self.best_x is not None and self.calls_made < self.budget - 10 and time.time() < self.deadline:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 50,
                            'maxls': 10,
                            'ftol': 1e-14,
                            'gtol': 1e-12,
                            'maxfun': 100
                        }
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass

        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)
        
        return self.best_x, self.best_f
