"""Root finding: Ill-conditioned - Bracket-first adaptive solver with Brent hybrid."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses bracket-first strategy with adaptive step-doubling to find sign change,
    then applies Brent's method (secant + bisection) for guaranteed convergence.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        return f(x)
    
    def time_ok():
        return time.time() - start_time < MAX_WALL
    
    def brent_hybrid(a, fa, b, fb, max_iters=80):
        """
        Brent's method variant: uses secant when safe, falls back to bisection.
        Requires fa and fb to have opposite signs (or one is zero).
        Returns (x, fx, success)
        """
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Track the best point
        best_x = a if abs(fa) < abs(fb) else b
        best_fx = fa if abs(fa) < abs(fb) else fb
        
        # Initialize c, d, e
        c = b
        fc = fb
        e = b - a
        d = e
        
        for _ in range(max_iters):
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            
            # Check if we've converged
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            
            # Update best
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            
            # Check if bracket is too small
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
            
            # Ensure fa and fb have opposite signs
            if fa * fb > 0:
                # This shouldn't happen if we maintain the bracket correctly
                break
            
            # Inverse quadratic interpolation
            if a != c and fa != fc:
                # Try IQUO
                s = fb / fa
                q = (fa - fc) / (c - a)
                p = fb * (fb - fa) / (2 * (fb - fc) if fb != fc else 1e-300)
                # Use the formula from Brent's paper
                if a != c:
                    p = (fb / fa) * ((fa - fc) / (c - a)) * (b - a) + (fb / fa) * (fb - fa)
                    # Simplified: use secant if IQUO is too complex
                    # Fall through to secant
                pass
            
            # Secant step
            if a != b:
                s = fb / fa
                # Secant formula
                x_new = a - fa * (b - a) / (fb - fa) if fb != fa else (a + b) / 2
            else:
                x_new = (a + b) / 2
            
            # Check if secant step is acceptable
            cond1 = (x_new - a) < 0.5 * (c - a) and (x_new - a) > 0
            cond2 = (a != c) and (abs(fb) < abs(fa))
            
            if cond1 and cond2:
                # Use secant
                e = d
                d = x_new - a
            else:
                # Use bisection
                d = 0.5 * (b - a)
            
            x_new = a + d
            
            # Ensure x_new is in the bracket
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2
            
            if x_new == a or x_new == b:
                break
            
            if eval_count[0] >= MAX_EVALS:
                break
            
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new, fx_new, True
            
            # Update bracket
            if fa * fx_new <= 0:
                b, fb = x_new, fx_new
            else:
                c, fc = a, fa
                a, fa = x_new, fx_new
            
            # Update best
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
        
        return best_x, best_fx, abs(best_fx) <= TARGET_TOL
    
    def find_bracket(x0, fx0):
        """
        Use step-doubling to find a bracket [a, b] where f(a) and f(b) have opposite signs.
        Returns (a, fa, b, fb, found_bracket)
        """
        # Start with small step
        h = 1e-6 * max(1.0, abs(x0))
        best_x = x0
        best_fx = fx0
        
        for scale in range(0, 30):  # Up to 2^30 steps
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
            
            h = 1e-6 * max(1.0, abs(x0)) * (2 ** scale)
            if h < 1e-15:
                continue
            
            for sign in [1, -1]:
                if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                    break
                
                x_c = x0 + sign * h
                if x_c == x0:
                    continue
                
                if eval_count[0] >= MAX_EVALS:
                    break
                
                fx_c = eval_f(x_c)
                
                if abs(fx_c) <= TARGET_TOL:
                    return x_c, fx_c, x_c, fx_c, True
                
                if abs(fx_c) < abs(best_fx):
                    best_x, best_fx = x_c, fx_c
                
                # Check for sign change with x0
                if fx0 * fx_c < 0:
                    a, b = (x0, x_c) if x0 < x_c else (x_c, x0)
                    fa, fb = (fx0, fx_c) if x0 < x_c else (fx_c, fx0)
                    return a, fa, b, fb, True
        
        return best_x, best_fx, best_x, best_fx, False
    
    def polish(x, fx, max_iters=20):
        """
        Final polish using small secant steps to squeeze out last digits of precision.
        """
        if abs(fx) <= TARGET_TOL:
            return x, fx, True
        
        best_x = x
        best_fx = fx
        
        for _ in range(max_iters):
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(fx) <= TARGET_TOL:
                return x, fx, True
            
            # Try small secant step
            h = 1e-8 * max(1.0, abs(x))
            x2 = x + h
            if x2 == x:
                break
            
            if eval_count[0] >= MAX_EVALS:
                break
            
            fx2 = eval_f(x2)
            
            if abs(fx2) <= TARGET_TOL:
                return x2, fx2, True
            
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            # Secant step
            if fx != fx2:
                x_new = x - fx * (x2 - x) / (fx2 - fx)
                if math.isfinite(x_new) and x_new != x:
                    if eval_count[0] >= MAX_EVALS:
                        break
                    fx_new = eval_f(x_new)
                    if abs(fx_new) <= TARGET_TOL:
                        return x_new, fx_new, True
                    if abs(fx_new) < abs(best_fx):
                        best_x, best_fx = x_new, fx_new
                    x, fx = x_new, fx_new
                else:
                    break
            else:
                break
        
        return best_x, best_fx, abs(best_fx) <= TARGET_TOL
    
    # --- Main logic ---
    
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    best_x = x0
    best_fx = fx0
    
    # Step 1: Try to find a bracket using step-doubling
    a, fa, b, fb, found_bracket = find_bracket(x0, fx0)
    
    if found_bracket:
        # Step 2: Apply Brent hybrid on the bracket
        x_brent, fx_brent, ok = brent_hybrid(a, fa, b, fb, max_iters=60)
        if ok:
            # Step 3: Polish for final precision
            x_final, fx_final, ok_final = polish(x_brent, fx_brent, max_iters=15)
            if ok_final:
                return x_final
            return x_final
        
        if abs(fx_brent) < abs(best_fx):
            best_x, best_fx = x_brent, fx_brent
    
    # Step 3: Polish the best point found
    if time_ok() and eval_count[0] < MAX_EVALS - 5 and abs(best_fx) > 0:
        x_polished, fx_polished, ok_polish = polish(best_x, best_fx, max_iters=20)
        if ok_polish:
            return x_polished
        if abs(fx_polished) < abs(best_fx):
            best_x, best_fx = x_polished, fx_polished
    
    # Step 4: If we didn't find a bracket, try a few more strategies
    if not found_bracket and time_ok() and eval_count[0] < MAX_EVALS - 20:
        # Try secant from x0 with a nearby point
        h_sec = 1e-4 * max(1.0, abs(x0))
        x2 = x0 + h_sec
        if x2 != x0:
            if eval_count[0] >= MAX_EVALS:
                return best_x
            fx2 = eval_f(x2)
            if abs(fx2) <= TARGET_TOL:
                return x2
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            # Secant iteration
            x1, fx1 = x0, fx0
            for _ in range(30):
                if not time_ok() or eval_count[0] >= MAX_EVALS:
                    break
                if abs(fx1) <= TARGET_TOL:
                    return x1
                if abs(fx2) <= TARGET_TOL:
                    return x2
                if fx1 == fx2 or x1 == x2:
                    break
                d = (fx2 - fx1) / (x2 - x1)
                if d == 0 or not math.isfinite(d):
                    break
                x_new = x1 - fx1 / d
                if not math.isfinite(x_new):
                    break
                x1, fx1 = x2, fx2
                if eval_count[0] >= MAX_EVALS:
                    break
                x2, fx2 = x_new, eval_f(x_new)
            
            best_x = x1 if abs(fx1) <= abs(fx2) else x2
            best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
    
    # Final polish
    if time_ok() and eval_count[0] < MAX_EVALS - 5 and abs(best_fx) > 0:
        x_final, fx_final, _ = polish(best_x, best_fx, max_iters=10)
        if abs(fx_final) < abs(best_fx):
            best_x, best_fx = x_final, fx_final
    
    return best_x

# EVOLVE-BLOCK-END
