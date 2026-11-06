"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    max_evals = 500
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
    
    # Try Newton's method with adaptive step
    def try_newton(x0):
        x = float(x0)
        for i in range(80):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= 1e-10:
                return True
            # Central difference derivative
            h = 1e-8 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
                fm = call_f(x - h)
            except Exception:
                return False
            d = (fp - fm) / (2 * h)
            if abs(d) < 1e-30:
                return False
            step = fx / d
            # Damped Newton
            if abs(step) > 1.0:
                step = math.copysign(1.0, step)
            x_new = x - step
            # Check for divergence
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15 * max(1.0, abs(x)):
                return False
            x = x_new
        return False
    
    # Try secant method
    def try_secant(x0):
        x1 = float(x0)
        x2 = x1 + 1e-4 * max(1.0, abs(x1))
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        for i in range(80):
            if not check_time():
                return False
            if abs(f2) <= 1e-10:
                return True
            if abs(f1) <= 1e-10:
                return True
            if abs(f2 - f1) < 1e-30:
                return False
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new):
                return False
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        return False
    
    # Try bisection with bracket expansion
    def try_bisection(x0):
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= 1e-10:
            return True
        
        # Try to find a bracket
        step = 1e-4 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None
        
        for direction in [1, -1]:
            for i in range(20):
                try:
                    x_test = x + direction * step * (2 ** i)
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
                        break
                else:
                    if fb is None:
                        b, fb = x, fx
                    if fb * f_test < 0:
                        a, fa = x_test, f_test
                        break
        
        if a is None or b is None:
            return False
        
        for i in range(100):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= 1e-10:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                break
        
        return abs(fa) <= 1e-10 or abs(fb) <= 1e-10
    
    # Try methods in order of expected efficiency
    if try_newton(x0):
        return best_x
    if try_secant(x0):
        return best_x
    if try_bisection(x0):
        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
