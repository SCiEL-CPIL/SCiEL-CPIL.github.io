# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Sobol-grid global search + Nelder-Mead local refinement on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.max_calls = self.budget - 5  # Safety margin
        self._deadline = None

    def __call__(self, func):
        start = time.time()
        self._deadline = start + 290.0  # 300s limit, keep 10s buffer
        d = self.dim
        rng = np.random.default_rng(42)

        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call(x):
            if self._calls >= self.max_calls:
                return None
            self._calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if np.isfinite(val):
                    return float(val)
                else:
                    return 1e10
            except Exception:
                return 1e10

        def time_ok():
            return time.time() < self._deadline

        # ---------------- Phase A: Sobol grid for global basin discovery ----------------
        # Use a good number of initial points to cover the space
        n_sobol = 32
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_pts = sampler.random(n_sobol)
        except Exception:
            sobol_pts = rng.random((n_sobol, d))
        
        sobol_pts = sobol_pts * 10.0 - 5.0
        sobol_pts = np.clip(sobol_pts, -5.0, 5.0)

        candidates = []
        for pt in sobol_pts:
            if not time_ok() or self._calls >= self.max_calls:
                break
            fx = call(pt)
            if fx is None:
                break
            candidates.append((fx, pt.copy()))
            if fx < self._best_f:
                self._best_f, self._best_x = fx, pt.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # ---------------- Phase B: Nelder-Mead local refinement ----------------
        # Run NM on the top 5 best initial points
        top_k = min(5, len(candidates))
        
        for i in range(top_k):
            if not time_ok() or self._calls >= self.max_calls:
                break
            x0 = candidates[i][1]
            self._nelder_mead(func, call, x0, time_ok)

        # ---------------- Phase C: Final Polish ----------------
        # If we have budget left, try to refine the best point further
        # Use a tighter tolerance or a different method if available
        if self._best_x is not None and self._calls < self.max_calls and time_ok():
            # Try a few steps of a simple coordinate descent or just re-run NM with tighter tol
            # NM is usually sufficient, but let's try to squeeze more out if we have lots of budget left
            if self._calls < self.max_calls - 100:
                 # Re-run NM from the best point found, maybe with a slightly perturbed start if stuck?
                 # Or just ensure we didn't stop early due to loose tolerance.
                 # The previous NM runs should have converged. 
                 # Let's try a simple local search around the best point with very small steps.
                 self._local_polish(call, self._best_x, time_ok)

        return self._best_x, self._best_f

    def _nelder_mead(self, func, call, x0, time_ok):
        d = self.dim
        
        # Simple Nelder-Mead implementation to control call counts precisely
        # SciPy's minimize with 'Nelder-Mead' might not expose call count easily in all versions or might overshoot
        # Implementing a lightweight NM is safer for budget control.
        
        # Initialize simplex
        x0 = x0.copy()
        f0 = call(x0)
        if f0 is None:
            return
        
        if f0 < self._best_f:
            self._best_f, self._best_x = f0, x0.copy()

        # Build simplex
        simplex = [x0]
        func_vals = [f0]
        
        # Standard NM initialization
        for i in range(d):
            xi = x0.copy()
            # Step size: 5% of the box size (10.0)
            step = 0.5 if i == 0 else 0.05 * 10.0 
            # Actually, standard is: if x0[i] != 0, step = 0.05 * x0[i], else 0.0025
            # But for box [-5,5], let's use a fixed reasonable step
            step = 0.5
            xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            
            if np.allclose(xi, x0):
                # If step was too small or clipped to same point, try opposite
                xi[i] -= 2 * step
                xi = np.clip(xi, -5.0, 5.0)
                
            fi = call(xi)
            if fi is None:
                return
            
            simplex.append(xi)
            func_vals.append(fi)
            
            if fi < self._best_f:
                self._best_f, self._best_x = fi, xi.copy()

        # NM Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        max_iter = 100
        tol = 1e-8
        
        for _ in range(max_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break
                
            # Sort simplex by function value
            order = np.argsort(func_vals)
            simplex = [simplex[i] for i in order]
            func_vals = [func_vals[i] for i in order]
            
            # Check convergence
            if abs(func_vals[0] - func_vals[-1]) < tol:
                break
            if np.linalg.norm(np.array(func_vals) - func_vals[0]) < tol:
                break

            x_best = simplex[0]
            f_best = func_vals[0]
            x_worst = simplex[-1]
            f_worst = func_vals[-1]
            x_second_worst = simplex[-2]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = call(x_ref)
            if f_ref is None:
                return
            
            if f_ref < self._best_f:
                self._best_f, self._best_x = f_ref, x_ref.copy()

            if f_ref < f_best:
                # Expansion
                x_exp = centroid + gamma * (centroid - x_worst)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = call(x_exp)
                if f_exp is None:
                    return
                
                if f_exp < self._best_f:
                    self._best_f, self._best_x = f_exp, x_exp.copy()
                    
                if f_exp < f_ref:
                    x_new, f_new = x_exp, f_exp
                else:
                    x_new, f_new = x_ref, f_ref
            elif f_ref < func_vals[-2]:
                # Accept reflection
                x_new, f_new = x_ref, f_ref
            else:
                # Contraction
                # Outside contraction if f_ref < f_worst, else inside
                if f_ref < f_worst:
                    x_con = centroid + rho * (x_worst - centroid) # This is actually inside contraction formula? 
                    # Standard: x_con = centroid - rho * (centroid - x_worst) for inside
                    # Let's use standard inside contraction
                    x_con = centroid - rho * (centroid - x_worst)
                else:
                    x_con = centroid + rho * (centroid - x_worst)
                    
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = call(x_con)
                if f_con is None:
                    return
                
                if f_con < self._best_f:
                    self._best_f, self._best_x = f_con, x_con.copy()

                if f_con < min(f_ref, f_worst):
                    x_new, f_new = x_con, f_con
                else:
                    # Shrinkage
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_i = call(simplex[i])
                        if f_i is None:
                            return
                        func_vals[i] = f_i
                        if f_i < self._best_f:
                            self._best_f, self._best_x = f_i, simplex[i].copy()
                    continue # Restart loop with shrunk simplex
            
            # Replace worst point
            simplex[-1] = x_new
            func_vals[-1] = f_new

    def _local_polish(self, call, x0, time_ok):
        # A very simple coordinate descent polish to clean up any remaining error
        # This is low cost and can improve precision if NM stopped slightly early
        d = self.dim
        x = x0.copy()
        f = call(x)
        if f is None:
            return
        
        if f < self._best_f:
            self._best_f, self._best_x = f, x.copy()

        step_sizes = [0.1, 0.01, 0.001, 0.0001]
        
        for step in step_sizes:
            if not time_ok() or self._calls >= self.max_calls:
                break
            improved = True
            while improved:
                improved = False
                for i in range(d):
                    if not time_ok() or self._calls >= self.max_calls:
                        return
                    
                    # Try positive step
                    xp = x.copy()
                    xp[i] += step
                    xp = np.clip(xp, -5.0, 5.0)
                    fp = call(xp)
                    if fp is None:
                        return
                    
                    if fp < f:
                        x = xp
                        f = fp
                        improved = True
                        if f < self._best_f:
                            self._best_f, self._best_x = f, x.copy()
                        continue # Restart loop for this step size as coordinates changed
                    
                    # Try negative step
                    xm = x.copy()
                    xm[i] -= step
                    xm = np.clip(xm, -5.0, 5.0)
                    fm = call(xm)
                    if fm is None:
                        return
                        
                    if fm < f:
                        x = xm
                        f = fm
                        improved = True
                        if f < self._best_f:
                            self._best_f, self._best_x = f, x.copy()

# EVOLVE-BLOCK-END
