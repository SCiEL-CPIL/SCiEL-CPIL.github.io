"""Combined root-finder: Broyden's method with finite-difference Jacobian,
backtracking line search, and Nelder-Mead fallback.
Uses a call counter to stay within budget and a time deadline for safety."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with finite-difference initial Jacobian and backtracking
       line search. This is efficient (quasi-Newton) and robust.
    2. If Broyden fails, try Nelder-Mead on ||F||^2 as a gradient-free fallback.
    
    Safety: time deadline and call budget to avoid failure.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 120s limit
    
    x = np.array(x0, dtype=float).copy()
    best_x = x.copy()
    best_norm = float('inf')
    calls = 0
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= 495:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None
    
    # Initial evaluation
    try:
        Fx = safe_F(x)
        if Fx is None:
            return best_x
        norm = np.linalg.norm(Fx)
        best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        return best_x
    
    # Compute initial Jacobian using finite differences
    h = 1e-6
    J = np.zeros((2, 2))
    jacobian_valid = True
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                jacobian_valid = False
                break
            J[:, j] = (Fx_j - Fx) / h
        except Exception:
            jacobian_valid = False
            break
    
    if not jacobian_valid:
        J = np.eye(2)
    
    # Broyden's method with line search
    max_broyden_iter = 100
    for _ in range(max_broyden_iter):
        if time.time() > deadline:
            break
        if calls >= 490:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Try pseudo-inverse or restart
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Backtracking line search
        alpha = 1.0
        step_accepted = False
        
        for _ in range(20):
            x_try = x + alpha * dx
            try:
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                # Accept if sufficient decrease
                if norm_try < norm * 0.9:
                    # Update Broyden's Jacobian estimate
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        # Step too small, reset Jacobian
                        J = np.eye(2)
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            except Exception:
                alpha *= 0.5
                continue
        
        if not step_accepted:
            break
        
        if norm <= 1e-8:
            return x
    
    # If Broyden didn't converge, try Nelder-Mead as fallback
    if norm > 1e-8 and time.time() < deadline and calls < 480:
        # Nelder-Mead on ||F||^2
        def objective(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError
            if calls >= 495:
                raise RuntimeError("Call limit exceeded")
            try:
                Fx = F(x)
                Fx = np.asarray(Fx, dtype=float)
                if Fx.shape != (2,):
                    Fx = Fx.reshape(2)
                if not np.all(np.isfinite(Fx)):
                    return 1e10
                calls += 1
                return np.sum(Fx ** 2)
            except Exception:
                return 1e10
        
        x_nm = best_x.copy()
        delta = 0.1
        simplex = [x_nm.copy(),
                   x_nm + np.array([delta, 0.0]),
                   x_nm + np.array([0.0, delta])]
        
        try:
            f_vals = [objective(v) for v in simplex]
        except Exception:
            return best_x
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        max_nm_iter = 150
        
        for _ in range(max_nm_iter):
            if time.time() > deadline:
                break
            if calls >= 490:
                break
            
            # Sort simplex
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            if f_vals[0] <= 1e-16:  # ||F||^2 <= 1e-16 => ||F|| <= 1e-8
                return simplex[0]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            try:
                f_ref = objective(x_ref)
            except Exception:
                break
            
            if f_vals[0] <= f_ref < f_vals[-2]:
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            elif f_ref < f_vals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                try:
                    f_exp = objective(x_exp)
                except Exception:
                    break
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                try:
                    f_con = objective(x_con)
                except Exception:
                    break
                if f_con < f_vals[-1]:
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        try:
                            f_vals[i] = objective(simplex[i])
                        except Exception:
                            break
                    else:
                        continue
                    break
        
        # Check if Nelder-Mead found a root
        idx = np.argmin(f_vals)
        if f_vals[idx] <= 1e-16:
            return simplex[idx]
        if np.sqrt(f_vals[idx]) < best_norm:
            best_x = simplex[idx].copy()
            best_norm = np.sqrt(f_vals[idx])
    
    return best_x
# EVOLVE-BLOCK-END
