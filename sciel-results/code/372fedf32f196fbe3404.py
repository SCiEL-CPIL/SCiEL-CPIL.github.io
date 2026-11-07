"""Derivative-free Trust-Region BFGS optimizer for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DF-TBFGS: Derivative-free Trust-Region BFGS.
    - Uses central finite differences to estimate gradients.
    - Maintains a BFGS approximation of the Hessian.
    - Uses trust-region steps with adaptive radius.
    - Multi-start with local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 330s kill
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
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def estimate_gradient(x, h=1e-4):
            """Estimate gradient using central finite differences."""
            g = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                fx_plus = try_eval(x_plus)
                fx_minus = try_eval(x_minus)
                
                if fx_plus is None or fx_minus is None:
                    return None
                
                g[i] = (fx_plus - fx_minus) / (2 * h)
            return g

        def df_tbfgs(x0, fx0, max_iter=200):
            """
            Run DF-TBFGS from starting point x0 with value fx0.
            Returns (x_best, f_best) for this local optimization.
            """
            nonlocal best_x, best_f, calls_used
            
            # BFGS state
            H = np.eye(d)  # Hessian approximation (inverse)
            x = x0.copy()
            f = fx0
            
            # Trust region radius
            trust_r = 1.0
            min_trust_r = 1e-5
            max_trust_r = 5.0
            
            no_improve = 0
            max_no_improve = 15
            
            for _ in range(max_iter):
                if calls_used >= max_calls or time.time() > deadline - 5:
                    break
                
                # Estimate gradient at current point
                g = estimate_gradient(x)
                if g is None:
                    break
                
                # Compute BFGS search direction: p = -H @ g
                p = -H @ g
                
                # Trust region step
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-15:
                    # Gradient is nearly zero, converged
                    break
                
                # Scale step to trust region
                if p_norm > trust_r:
                    p = p * (trust_r / p_norm)
                
                x_new = x + p
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = try_eval(x_new)
                if fx_new is None:
                    break
                
                # Check improvement
                actual_reduction = f - fx_new
                predicted_reduction = -0.5 * (p @ g)  # linear model prediction
                
                rho = 0.0
                if predicted_reduction > 1e-12:
                    rho = actual_reduction / predicted_reduction
                
                if rho > 0.25:
                    # Good step: accept, possibly expand trust region
                    # BFGS update
                    s = x_new - x
                    y = estimate_gradient(x_new)
                    if y is not None:
                        sy = s @ y
                        if sy > 1e-10:
                            # BFGS update formula
                            Hy = H @ y
                            H = H + ((sy + s @ Hy) / (sy * sy)) * np.outer(s, s) - \
                                  (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                            # Ensure positive definiteness
                            eigvals = np.linalg.eigvalsh(H)
                            if np.min(eigvals) < 1e-6:
                                H = H + (1e-5 - np.min(eigvals)) * np.eye(d)
                    
                    x = x_new
                    f = fx_new
                    trust_r = min(trust_r * 1.5, max_trust_r)
                    no_improve = 0
                elif rho > 0.1:
                    # Marginal step: accept but don't expand
                    s = x_new - x
                    y = estimate_gradient(x_new)
                    if y is not None:
                        sy = s @ y
                        if sy > 1e-10:
                            Hy = H @ y
                            H = H + ((sy + s @ Hy) / (sy * sy)) * np.outer(s, s) - \
                                  (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                            eigvals = np.linalg.eigvalsh(H)
                            if np.min(eigvals) < 1e-6:
                                H = H + (1e-5 - np.min(eigvals)) * np.eye(d)
                    
                    x = x_new
                    f = fx_new
                    trust_r = trust_r  # keep same
                    no_improve = 0
                else:
                    # Poor step: reject, shrink trust region
                    trust_r = max(trust_r * 0.5, min_trust_r)
                    no_improve += 1
                    
                    if no_improve >= max_no_improve:
                        # Reset Hessian and try a random jump
                        H = np.eye(d)
                        jump = trust_r * 2 * rng.standard_normal(d)
                        x_jump = np.clip(x + jump, -5, 5)
                        fx_jump = try_eval(x_jump)
                        if fx_jump is None:
                            break
                        if fx_jump < f:
                            x = x_jump
                            f = fx_jump
                            trust_r = 0.5
                            no_improve = 0
                        else:
                            break
            return x, f

        # Phase 1: Multi-start with local refinement
        n_starts = 5
        start_points = [np.zeros(d)]
        for i in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, d))
        
        for sp in start_points:
            if calls_used >= max_calls or time.time() > deadline - 10:
                break
            
            fx_start = try_eval(sp)
            if fx_start is None:
                break
            
            if fx_start < best_f:
                best_f = fx_start
                best_x = sp.copy()
            
            # Run local DF-TBFGS from this start
            x_local, f_local = df_tbfgs(sp, fx_start)
            
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()
        
        # Phase 2: Polish the best solution with smaller steps
        if calls_used < max_calls and time.time() < deadline - 3 and best_x is not None:
            # Reset Hessian for polish
            H = np.eye(d)
            x = best_x.copy()
            f = best_f
            trust_r = 0.1
            min_trust_r = 1e-6
            max_no_improve = 8
            no_improve = 0
            
            for _ in range(100):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break
                
                g = estimate_gradient(x)
                if g is None:
                    break
                
                p = -H @ g
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-12:
                    break
                
                if p_norm > trust_r:
                    p = p * (trust_r / p_norm)
                
                x_new = np.clip(x + p, -5, 5)
                fx_new = try_eval(x_new)
                if fx_new is None:
                    break
                
                actual_reduction = f - fx_new
                predicted_reduction = -0.5 * (p @ g)
                
                rho = 0.0
                if predicted_reduction > 1e-12:
                    rho = actual_reduction / predicted_reduction
                
                if rho > 0.25:
                    s = x_new - x
                    y = estimate_gradient(x_new)
                    if y is not None:
                        sy = s @ y
                        if sy > 1e-10:
                            Hy = H @ y
                            H = H + ((sy + s @ Hy) / (sy * sy)) * np.outer(s, s) - \
                                  (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                            eigvals = np.linalg.eigvalsh(H)
                            if np.min(eigvals) < 1e-6:
                                H = H + (1e-5 - np.min(eigvals)) * np.eye(d)
                    
                    x = x_new
                    f = fx_new
                    trust_r = min(trust_r * 1.3, 0.5)
                    no_improve = 0
                    
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                elif rho > 0.1:
                    s = x_new - x
                    y = estimate_gradient(x_new)
                    if y is not None:
                        sy = s @ y
                        if sy > 1e-10:
                            Hy = H @ y
                            H = H + ((sy + s @ Hy) / (sy * sy)) * np.outer(s, s) - \
                                  (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                            eigvals = np.linalg.eigvalsh(H)
                            if np.min(eigvals) < 1e-6:
                                H = H + (1e-5 - np.min(eigvals)) * np.eye(d)
                    
                    x = x_new
                    f = fx_new
                    no_improve = 0
                    
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    trust_r = max(trust_r * 0.5, min_trust_r)
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
