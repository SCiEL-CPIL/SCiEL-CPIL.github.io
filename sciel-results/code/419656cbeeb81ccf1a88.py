"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using:
    1. Success-rate-adaptive random walk for global exploration
    2. Nelder-Mead simplex for local refinement
    3. Multiple restarts from diverse points
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: Success-rate-adaptive random walk ---
        # Start from origin and a few random points
        x = np.zeros(dim)
        fx = safe_call(x)
        
        # Adaptive step size based on acceptance rate
        step = 2.0
        window = 20
        accept_count = 0
        total_count = 0
        
        max_walk_steps = budget // 2  # Use half budget for global walk
        
        for i in range(max_walk_steps):
            if calls_made >= budget or time.time() > deadline:
                break
            
            direction = rng.standard_normal(dim)
            y = np.clip(x + step * direction, -5.0, 5.0)
            fy = safe_call(y)
            
            if fy < fx:
                x, fx = y, fy
                accept_count += 1
            
            total_count += 1
            
            # Adapt step size every 'window' evaluations
            if total_count % window == 0:
                rate = accept_count / total_count
                if rate > 0.5:
                    step *= 1.3
                elif rate < 0.2:
                    step *= 0.7
                step = np.clip(step, 0.01, 5.0)
                accept_count = 0
                total_count = 0
        
        # --- Phase 2: Nelder-Mead from best point and a few neighbors ---
        nm_points = [best_x.copy() if best_x is not None else np.zeros(dim)]
        # Add a few perturbed points for diversity
        if best_x is not None:
            for scale in [0.5, 1.0, 2.0]:
                nm_points.append(np.clip(best_x + rng.normal(0, scale, dim), -5.0, 5.0))
        
        for start_pt in nm_points:
            if calls_made >= budget or time.time() > deadline:
                break
            
            x_opt, f_opt = self._nelder_mead(func, safe_call, start_pt, budget, deadline, dim, rng)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # --- Phase 3: Final fine-tuning with small random steps ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            for _ in range(min(50, budget - calls_made)):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = np.clip(x + 0.1 * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx
        
        # Ensure at least one call was made
        if calls_made == 0:
            safe_call(np.zeros(dim))
        
        return best_x, best_f

    def _nelder_mead(self, func, safe_call, x0, budget, deadline, dim, rng):
        """Nelder-Mead simplex method with early termination."""
        x = x0.copy()
        fx = safe_call(x)
        
        # Build initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]
        
        for d in range(dim):
            xi = x.copy()
            xi[d] += 0.1 if x[d] + 0.1 <= 5.0 else -0.1
            simplex.append(np.clip(xi, -5.0, 5.0))
            simplex_f.append(safe_call(simplex[-1]))
        
        # Track best
        best_idx = int(np.argmin(simplex_f))
        best_x = simplex[best_idx].copy()
        best_f = simplex_f[best_idx]
        
        max_iter = min(200, (budget - calls_made) // (dim + 1))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                # Contraction
                rho = 0.5
                if fr < worst_f:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - worst)
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc < worst_f:
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = safe_call(simplex[i])
            
            # Update best
            cur_best_idx = int(np.argmin(simplex_f))
            if simplex_f[cur_best_idx] < best_f:
                best_f = simplex_f[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()
            
            # Check convergence
            if np.max(simplex_f) - np.min(simplex_f) < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
