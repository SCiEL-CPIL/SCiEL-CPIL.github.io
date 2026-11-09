import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B with Rosenbrock-aware valley initialization."""

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
        self.deadline = self.start_time + 280.0
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

        def rosenbrock_valley_point(a):
            """Generate a point on the Rosenbrock valley: x[0]=a, x[i]=a^(2^i)."""
            x = np.zeros(self.dim)
            x[0] = a
            for i in range(1, self.dim):
                x[i] = x[i-1] ** 2
            return np.clip(x, -5, 5)
        
        def efficient_valley_track(x0, max_iters=40):
            """
            Efficient valley tracking for Rosenbrock.
            Uses the known valley geometry x[i+1] ≈ x[i]^2 to guide search.
            Much cheaper than full coordinate descent.
            """
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 3:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Primary: track the valley by adjusting x[i+1] toward x[i]^2
                # This is the key structural feature of Rosenbrock
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 3:
                        break
                    target = x[i] ** 2
                    # Try to move x[i+1] closer to target
                    if abs(x[i+1] - target) > 0.01:
                        x_try = x.copy()
                        x_try[i+1] = 0.5 * x[i+1] + 0.5 * target
                        x_try = np.clip(x_try, -5, 5)
                        f_try = wrapped_func(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                
                # Secondary: fine-tune x[0] (the valley parameter)
                if not improved and self.dim > 0:
                    for delta in [0.05, -0.05, 0.02, -0.02, 0.1, -0.1]:
                        if self.calls_made >= self.budget - 3:
                            break
                        x_try = x.copy()
                        x_try[0] += delta
                        x_try = np.clip(x_try, -5, 5)
                        f_try = wrapped_func(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                            break
                
                # Tertiary: small perturbations if stuck
                if not improved:
                    rng_local = np.random.default_rng(hash(tuple(x.tolist())) % 2**32)
                    x_try = x + rng_local.normal(0, 0.05, self.dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = wrapped_func(x_try)
                    if f_try < f:
                        x = x_try
                        f = f_try
                        improved = True
                    else:
                        break
                
            return x, f

        def valley_follow(x0, max_iters=30):
            """
            Adaptive valley-following for Rosenbrock.
            Exploits the known geometry: the valley follows x[i+1] ≈ x[i]^2.
            Uses coordinate-wise steps with adaptive sizing.
            """
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            step_size = 0.3
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Valley-following: pull x[i+1] toward x[i]^2
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 5:
                        break
                    target_next = x[i] ** 2
                    # Try multiple blend strengths
                    for blend in [0.5, 0.7, 0.9]:
                        x_vf = x.copy()
                        x_vf[i+1] = (1 - blend) * x[i+1] + blend * target_next
                        x_vf = np.clip(x_vf, -5, 5)
                        f_vf = wrapped_func(x_vf)
                        if f_vf < f:
                            x = x_vf
                            f = f_vf
                            improved = True
                            break
                    # If no blend helped, try small adjustments
                    if not improved:
                        for delta in [0.1, -0.1, 0.05, -0.05]:
                            if self.calls_made >= self.budget - 5:
                                break
                            x_adj = x.copy()
                            x_adj[i+1] += delta
                            x_adj = np.clip(x_adj, -5, 5)
                            f_adj = wrapped_func(x_adj)
                            if f_adj < f:
                                x = x_adj
                                f = f_adj
                                improved = True
                                break
                
                # Coordinate descent with adaptive steps
                for i in range(self.dim):
                    if self.calls_made >= self.budget - 5:
                        break
                    best_local_f = f
                    best_local_x = x.copy()
                    for step_mult in [step_size, step_size * 0.5, step_size * 2.0]:
                        if self.calls_made >= self.budget - 5:
                            break
                        for direction in [1.0, -1.0]:
                            if self.calls_made >= self.budget - 5:
                                break
                            x_new = x.copy()
                            x_new[i] += step_mult * direction
                            x_new = np.clip(x_new, -5, 5)
                            f_new = wrapped_func(x_new)
                            if f_new < best_local_f:
                                best_local_f = f_new
                                best_local_x = x_new
                    if best_local_f < f:
                        x = best_local_x
                        f = best_local_f
                        improved = True
                        step_size = min(step_size * 1.1, 1.5)
                    else:
                        step_size = max(step_size * 0.7, 0.01)
                
                if not improved:
                    step_size = max(step_size * 0.5, 0.005)
                    if step_size < 0.01:
                        break
            
            return x, f

        # === PHASE 1: Focused initialization using Rosenbrock geometry ===
        rng = np.random.default_rng(2024)
        
        # Generate valley points for key 'a' values (fewer, more targeted)
        a_values = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0]
        
        # Evaluate valley points
        valley_points = []
        for a in a_values:
            if self.calls_made >= self.budget - 20:
                break
            if time.time() > self.deadline:
                break
            vp = rosenbrock_valley_point(a)
            f_vp = wrapped_func(vp)
            valley_points.append((f_vp, vp.copy()))
        
        # Evaluate a few random points for diversity
        for _ in range(5):
            if self.calls_made >= self.budget - 20:
                break
            if time.time() > self.deadline:
                break
            rp = rng.uniform(-3, 3, self.dim)
            f_rp = wrapped_func(rp)
            valley_points.append((f_rp, rp.copy()))
        
        # Sort by function value
        valley_points.sort(key=lambda t: t[0])
        
        # Update best
        if valley_points:
            best_f_init, best_x_init = valley_points[0]
            if best_f_init < self.best_f:
                self.best_f = best_f_init
                self.best_x = best_x_init.copy()
        
        # === PHASE 2: Efficient optimization from good starts ===
        starts = []
        
        # Top valley points (fewer to save budget for refinement)
        for f_v, x_v in valley_points[:5]:
            starts.append(x_v.copy())
        
        # Perturbations of the best point
        if self.best_x is not None:
            for noise_scale in [0.2, 0.5]:
                for _ in range(2):
                    s = self.best_x + rng.normal(0, noise_scale, self.dim)
                    s = np.clip(s, -5, 5)
                    starts.append(s)
        
        # Additional valley points near the best
        if self.best_x is not None:
            a_best = self.best_x[0]
            for da in [-0.3, 0.3, -0.6, 0.6]:
                a_new = a_best + da
                vp = rosenbrock_valley_point(a_new)
                starts.append(vp)
        
        # Deduplicate
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=0.05):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget - 30:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 20:
                continue
            
            # Use Nelder-Mead with adaptive parameters (faster than L-BFGS-B for this problem)
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'maxfev': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                break
            except Exception:
                pass
            
            # Efficient valley tracking refinement
            if self.best_x is not None and self.calls_made < self.budget - 15:
                try:
                    x_vf, f_vf = efficient_valley_track(self.best_x, max_iters=25)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # === PHASE 3: High-precision polish ===
        if self.best_x is not None and self.calls_made < self.budget - 10 and time.time() < self.deadline:
            # Final Nelder-Mead with tighter tolerances
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': 150,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                pass
            except Exception:
                pass
            
            # Final efficient valley tracking polish
            if self.best_x is not None and self.calls_made < self.budget - 5:
                try:
                    x_vf, f_vf = efficient_valley_track(self.best_x, max_iters=30)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    pass
                except Exception:
                    pass
        
        # Fallback
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)
        
        return self.best_x, self.best_f
