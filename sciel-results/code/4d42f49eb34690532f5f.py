"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq, minimize

def find_root(f, x0):
    """
    Find a root of f using:
    1. Bracketing and Brent's method.
    2. Global optimization of f(x)^2 using Nelder-Mead/Powell if bracketing fails.
    """
    x0 = float(x0)
    eval_count = 0
    max_evals = 500
    target_tol = 1e-8

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Evaluation limit exceeded")
        eval_count += 1
        return float(f(x))

    # 1. Initial evaluation
    fx0 = eval_f(x0)
    if abs(fx0) <= target_tol:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # 2. Try to find a bracket
    h = 1e-6 * max(1.0, abs(x0))
    
    # We keep track of points evaluated to potentially use as starting points for optimization
    candidates = [(x0, fx0)]

    for i in range(100):
        if eval_count >= max_evals - 20:
            break
            
        a = x0 - h
        b = x0 + h
        
        try:
            fa = eval_f(a)
            candidates.append((a, fa))
            if abs(fa) < best_f_abs:
                best_f_abs = abs(fa)
                best_x = a
            if abs(fa) <= target_tol:
                return a
        except:
            break

        try:
            fb = eval_f(b)
            candidates.append((b, fb))
            if abs(fb) < best_f_abs:
                best_f_abs = abs(fb)
                best_x = b
            if abs(fb) <= target_tol:
                return b
        except:
            break

        # Check for bracket
        if fa * fb < 0:
            try:
                root = brentq(eval_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if abs(eval_f(root)) <= target_tol:
                    return root
                return root
            except Exception:
                pass

        h *= 2.0
        if h > 1e10:
            break

    # 3. Global optimization fallback
    # Use the best point found and potentially others as starting points
    # We will try Nelder-Mead and Powell from the best point
    
    def objective(x):
        return eval_f(x[0]) ** 2

    best_result_x = best_x
    best_result_val = best_f_abs

    # Sort candidates by |f(x)| to prioritize good starting points
    candidates.sort(key=lambda item: abs(item[1]))
    
    # Try top few candidates
    # Limit to 3 starting points to save evaluations
    start_points = [item[0] for item in candidates[:3]]
    
    # Ensure x0 is in there
    if x0 not in start_points:
        start_points.append(x0)

    for x_start in start_points:
        if eval_count >= max_evals - 10:
            break
            
        # Try Nelder-Mead with a large initial simplex
        try:
            simplex = np.array([[x_start], [x_start + 1.0 * max(1.0, abs(x_start))]])
            res_nm = minimize(objective, x0=np.array([x_start]), method='Nelder-Mead',
                              options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 100, 'maxfev': 100,
                                       'initial_simplex': simplex})
            if res_nm.success:
                x_opt = res_nm.x[0]
                f_opt = eval_f(x_opt)
                if abs(f_opt) < best_result_val:
                    best_result_val = abs(f_opt)
                    best_result_x = x_opt
                if abs(f_opt) <= target_tol:
                    return x_opt
        except Exception:
            pass

        # Try Powell
        try:
            res_pw = minimize(objective, x0=np.array([x_start]), method='Powell',
                              options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 100, 'maxfev': 100})
            if res_pw.success:
                x_opt = res_pw.x[0]
                f_opt = eval_f(x_opt)
                if abs(f_opt) < best_result_val:
                    best_result_val = abs(f_opt)
                    best_result_x = x_opt
                if abs(f_opt) <= target_tol:
                    return x_opt
        except Exception:
            pass

    return best_result_x
# EVOLVE-BLOCK-END
