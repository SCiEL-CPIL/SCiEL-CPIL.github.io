"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 135.0
    max_evals = 490
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def try_secant(x0, tol=TOL, max_iter=180):
        x1 = float(x0)
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            denom = f2 - f1
            if abs(denom) < 1e-30:
                return False
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            # Adaptive damping: use smaller max step when function values are large
            scale = max(1.0, abs(x2))
            max_step = 10.0 * scale
            if abs(f2) > 1.0:
                max_step = 5.0 * scale
            if abs(x_new - x2) > max_step:
                 x_new = x2 + math.copysign(max_step, x_new - x2)

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        return abs(f2) <= tol or abs(f1) <= tol

    def polish(x_start, tol=TOL, max_iter=80):
        """High-precision polish pass using secant with tighter tolerance."""
        x = float(x_start)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True
        
        h = 1e-7 * max(1.0, abs(x))
        x2 = x + h
        try:
            f2 = call_f(x2)
        except Exception:
            return False
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(fx) <= tol:
                return True
            denom = f2 - fx
            if abs(denom) < 1e-30:
                return False
            x_new = x2 - f2 * (x2 - x) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            # Gentle damping to avoid overshooting in ill-conditioned regions
            step = x_new - x2
            if abs(step) > 0.5 * max(1.0, abs(x2)):
                x_new = x2 + math.copysign(0.5 * max(1.0, abs(x2)), step)

            x, fx = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        return abs(f2) <= tol or abs(fx) <= tol

    def try_brent_like(x0, tol=TOL):
        # A simplified Brent's method logic focusing on interpolation when possible
        # but falling back to bisection.
        # We need a bracket first.
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True

        step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None

        # Search for bracket - more aggressive with larger initial step and more iterations
        found_bracket = False
        for direction in [1, -1]:
            for i in range(40):
                x_test = x + direction * step
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if direction > 0:
                    if fa is None:
                        a, fa = x, fx
                    if fa * f_test < 0:
                        b, fb = x_test, f_test
                        found_bracket = True
                        break
                    a, fa = x_test, f_test
                else:
                    if fb is None:
                        b, fb = x, fx
                    if fb * f_test < 0:
                        a, fa = x_test, f_test
                        found_bracket = True
                        break
                    b, fb = x_test, f_test
                step *= 1.8
        
        if not found_bracket:
            return False

        # Now solve in [a, b]
        c = a
        fc = fa
        d = 0.0
        e = 0.0
        
        for i in range(200):
            if not check_time():
                return False
            if abs(fb) <= tol:
                return True
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                return min(abs(fa), abs(fb)) <= tol

            if (fa != fc) and (fb != fc):
                s = fc
                fc = fa
                a = c
                c = s
            if abs(fb - fa) < abs(fa - fc):
                c = a
                fc = fa
                e = d = b - a
            else:
                s = d
                d = e = b - a
            
            m = (fa - fb) / (a - b)
            p = 2.0 * e * (fc - fb) - (b - a) * (fb - fc)
            
            if abs(p) < abs(e) * m:
                p = -math.copysign(max(abs(e) * m, 1e-15), e)
                e = d = b - a
            else:
                e = d = b - a
            
            if b - a < 1e-14 * max(1.0, abs(b)):
                break
            
            try:
                f_new = call_f(b + p)
            except Exception:
                return False
            
            if abs(f_new) <= tol:
                return True
            
            if fa * f_new < 0:
                c = b
                fc = fb
            else:
                c = a
                fc = fa
                a = b
                fa = fb
            b = b + p
            fb = f_new
            
        return min(abs(fa), abs(fb)) <= tol

    def try_newton_forward(x0, tol=TOL, max_iter=150):
        x = float(x0)
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            
            h = 1e-7 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
            except Exception:
                return False
            
            d = (fp - fx) / h
            if abs(d) < 1e-30:
                return False
            
            step = fx / d
            # Damp step if too large
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x = x_new
        return False

    # Strategy:
    # 1. Secant is very fast and often converges for well-behaved functions.
    # 2. If Secant fails or diverges, Brent's method (with bracket search) is robust.
    # 3. Newton with forward diff is a backup, but uses 2 evals/iter so it's slower in budget terms.
    # 4. Final polish pass with tighter tolerance to squeeze out residual error.

    solved = False
    if try_secant(x0):
        solved = True
    
    if not solved and try_brent_like(x0):
        solved = True
    
    if not solved and try_newton_forward(x0):
        solved = True
    
    # Final polish pass: if we have a good candidate, try to refine it
    if best_fabs > TOL and best_fabs < 1e-2:
        if polish(best_x, TOL, 80):
            solved = True
    
    # Ultimate high-precision refinement using inverse quadratic interpolation
    if best_fabs > TOL and best_fabs < 1e-4 and eval_count < max_evals - 10:
        x = best_x
        try:
            fx = call_f(x)
        except Exception:
            fx = best_fabs
        if abs(fx) > TOL:
            # Try a very small perturbation to get another point
            h = 1e-8 * max(1.0, abs(x))
            x2 = x + h
            try:
                f2 = call_f(x2)
            except Exception:
                f2 = fx
            if abs(f2) <= TOL:
                best_x = x2
                best_fabs = abs(f2)
                solved = True
            elif abs(fx) <= TOL:
                solved = True
            else:
                # Inverse quadratic interpolation step
                denom = (f2 - fx)
                if abs(denom) > 1e-30:
                    x_new = x2 - f2 * (x2 - x) / denom
                    if math.isfinite(x_new) and abs(x_new) < 1e15:
                        # Damp if step is large
                        if abs(x_new - x2) > 0.1 * max(1.0, abs(x2)):
                            x_new = x2 + math.copysign(0.1 * max(1.0, abs(x2)), x_new - x2)
                        try:
                            f_new = call_f(x_new)
                        except Exception:
                            f_new = best_fabs
                        if abs(f_new) < best_fabs:
                            best_x = x_new
                            best_fabs = abs(f_new)
                        if abs(f_new) <= TOL:
                            solved = True
        else:
            solved = True
    
    return best_x
# EVOLVE-BLOCK-END
