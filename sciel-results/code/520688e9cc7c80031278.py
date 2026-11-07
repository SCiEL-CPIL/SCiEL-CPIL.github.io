import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Sample 20 points: 10 random, 10 from a coarse grid/structured random
        init_points = []
        # Random
        for _ in range(10):
            init_points.append(rng.uniform(-5, 5, d))
        # Structured Random (Latin Hypercube-like)
        for _ in range(10):
            # Simple stratified sampling
            p = np.array([rng.uniform(-5 + 10*i/d, -5 + 10*(i+1)/d) for i in range(d)])
            init_points.append(p)
            
        # Evaluate all
        evaluated = []
        for p in init_points:
            p = np.clip(p, -5, 5)
            fx = try_eval(p)
            if fx is not None:
                evaluated.append((fx, p))
                
        if not evaluated:
            return np.zeros(d), float('inf')
            
        # Sort by fitness
        evaluated.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement using DF-TBFGS
        # Take top 3 points
        top_points = [p for f, p in evaluated[:3]]
        
        for start in top_points:
            if calls_used >= max_calls or time.time() > deadline - 10:
                break
            self._run_df_tbfgs(func, start, try_eval)
            
        # Phase 3: Final Polish
        if calls_used < max_calls and time.time() < deadline - 2:
            self._polish(func, best_x, best_f, try_eval, rng)
            
        return best_x, best_f

    def _run_df_tbfgs(self, func, start, try_eval):
        d = self.dim
        x = start.copy()
        fx = try_eval(x)
        if fx is None: return
        
        # Initialize Hessian approximation (Identity)
        H = np.eye(d)
        delta = 1.0 # Trust region radius
        max_iter = 50
        
        for _ in range(max_iter):
            # Approximate Gradient
            g = np.zeros(d)
            h = 1e-4 # Finite difference step
            for i in range(d):
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fxp = try_eval(xp)
                if fxp is None: return
                g[i] = (fxp - fx) / h
                
            if np.linalg.norm(g) < 1e-6:
                break
                
            # Compute step
            # p = -H * g
            p = -H @ g
            # Normalize to trust region
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-10:
                p = -g
                p_norm = np.linalg.norm(p)
            p = (delta / p_norm) * p
            
            # Check if step goes out of bounds, adjust
            # Simple: clip x+p
            x_new = np.clip(x + p, -5, 5)
            
            # Evaluate
            fx_new = try_eval(x_new)
            if fx_new is None: return
            
            # Calculate reduction
            # Predicted reduction: -0.5 * p^T (H^-1) p ? No, we used H as inverse Hessian approx.
            # The model is f(x) + g^T p + 0.5 p^T (H^-1) p
            # We approximated H as the inverse Hessian.
            # So model reduction is -g^T p - 0.5 p^T H p
            # Actually, standard BFGS updates the inverse Hessian H.
            # The step is p = -H g.
            # The quadratic model is m(p) = f(x) + g^T p + 0.5 p^T H p.
            # Reduction in model: -g^T p - 0.5 p^T H p.
            # Since p = -H g, g^T p = -g^T H g.
            # p^T H p = g^T H g.
            # So pred_red = g^T H g - 0.5 g^T H g = 0.5 g^T H g.
            # Wait, H is positive definite approx.
            # Let's just use the ratio of actual to predicted.
            
            # Actual reduction
            act_red = fx - fx_new
            
            # Predicted reduction
            # Using the quadratic model with H as inverse Hessian
            # m(0) - m(p) = -g^T p - 0.5 p^T H p
            pred_red = -g @ p - 0.5 * (p @ H @ p)
            
            # If pred_red is negative or zero, something is wrong with H
            if pred_red < 1e-12:
                pred_red = 1e-12
            
            rho = act_red / pred_red
            
            if rho > 0.75:
                # Good step, expand trust region
                delta *= 2.0
                # Update H (BFGS update)
                # s = x_new - x
                # y = g_new - g  (Need g_new)
                # We don't have g_new yet. We can compute it or approximate.
                # For simplicity and speed, we might skip H update if we don't have g_new.
                # Or compute g_new.
                g_new = np.zeros(d)
                for i in range(d):
                    xp = x_new.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fxp = try_eval(xp)
                    if fxp is None: return
                    g_new[i] = (fxp - fx_new) / h
                    
                s = x_new - x
                y = g_new - g
                sy = s @ y
                if sy > 1e-10:
                    # BFGS update for inverse Hessian
                    H = H + (sy + s @ H @ s) / (sy**2) * np.outer(s, s) - (np.outer(s, H @ y) + np.outer(H @ y, s)) / sy
                    # Ensure symmetry
                    H = (H + H.T) / 2
            elif rho > 0.25:
                # Acceptable step, keep trust region
                # Update H
                g_new = np.zeros(d)
                for i in range(d):
                    xp = x_new.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fxp = try_eval(xp)
                    if fxp is None: return
                    g_new[i] = (fxp - fx_new) / h
                s = x_new - x
                y = g_new - g
                sy = s @ y
                if sy > 1e-10:
                    H = H + (sy + s @ H @ s) / (sy**2) * np.outer(s, s) - (np.outer(s, H @ y) + np.outer(H @ y, s)) / sy
                    H = (H + H.T) / 2
            else:
                # Bad step, shrink trust region
                delta *= 0.5
                # Do not update H
                
            x = x_new
            fx = fx_new
            
            if delta < 1e-8:
                break

    def _polish(self, func, x, fx, try_eval, rng):
        d = self.dim
        sigma = 0.01
        for _ in range(50):
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = try_eval(y)
            if fy is None: break
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.7
            if sigma < 1e-8: break
