import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    x = float(x0)

    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # History for quadratic/secant: list of (x, fx)
    history = [(x, fx)]
    
    # Parameters
    max_iter = 50

    for i in range(max_iter):
        if time.time() > deadline:
            break

        x_new = None

        # Strategy: 
        # 1. If we have 3 distinct points with distinct f values, use quadratic interpolation
        # 2. If we have 2 points, use Secant
        # 3. If we have 1 point, use Newton with FD derivative

        if len(history) >= 3:
            x0_h, f0_h = history[-3]
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            
            # Check that x values are sufficiently distinct
            if (abs(x2_h - x1_h) > 1e-10 and abs(x1_h - x0_h) > 1e-10 and abs(x2_h - x0_h) > 1e-10):
                # Check that f values are sufficiently distinct for quadratic interpolation
                if (abs(f1_h - f0_h) > 1e-12 and abs(f2_h - f1_h) > 1e-12 and abs(f2_h - f0_h) > 1e-12):
                    # Fit quadratic: f(x) = a + b*(x-x2) + c*(x-x2)^2
                    # Using divided differences:
                    # f[x0,x1] = (f1-f0)/(x1-x0)
                    # f[x1,x2] = (f2-f1)/(x2-x1)
                    # f[x0,x1,x2] = (f[x1,x2]-f[x0,x1])/(x2-x0)
                    f01 = (f1_h - f0_h) / (x1_h - x0_h)
                    f12 = (f2_h - f1_h) / (x2_h - x1_h)
                    f012 = (f12 - f01) / (x2_h - x0_h)
                    
                    # Quadratic in Newton form: f(x) = f2 + f12*(x-x2) + f012*(x-x2)*(x-x1)
                    # Root: f012*(x-x2)*(x-x1) + f12*(x-x2) + f2 = 0
                    # Let t = x - x2, then x - x1 = t + (x2 - x1)
                    # f012 * t * (t + (x2-x1)) + f12 * t + f2 = 0
                    # f012 * t^2 + f012*(x2-x1)*t + f12*t + f2 = 0
                    # f012 * t^2 + (f012*(x2-x1) + f12)*t + f2 = 0
                    
                    a_q = f012
                    b_q = f012 * (x2_h - x1_h) + f12
                    c_q = f2_h
                    
                    if abs(a_q) > 1e-14:
                        # Solve quadratic: a_q*t^2 + b_q*t + c_q = 0
                        disc = b_q * b_q - 4 * a_q * c_q
                        if disc >= 0:
                            sqrt_disc = math.sqrt(disc)
                            # Choose root that gives smaller |t| to avoid cancellation
                            # t = (-b_q ± sqrt_disc) / (2*a_q)
                            # Choose sign to avoid catastrophic cancellation
                            if abs(b_q) < 1e-14:
                                t1 = sqrt_disc / (2 * a_q)
                                t2 = -sqrt_disc / (2 * a_q)
                            elif b_q > 0:
                                t1 = (-b_q - sqrt_disc) / (2 * a_q)
                                t2 = -c_q / (a_q * t1) if t1 != 0 else 0
                            else:
                                t1 = (-b_q + sqrt_disc) / (2 * a_q)
                                t2 = -c_q / (a_q * t1) if t1 != 0 else 0
                            
                            # Pick the root with smaller |t| (closer to x2)
                            if abs(t1) <= abs(t2):
                                t = t1
                            else:
                                t = t2
                            
                            x_new = x2_h + t
                        else:
                            # No real root from quadratic, fall through to secant
                            x_new = None
                    else:
                        # Linear case (a_q ~ 0): b_q*t + c_q = 0
                        if abs(b_q) > 1e-14:
                            t = -c_q / b_q
                            x_new = x2_h + t
                        else:
                            x_new = None
        
        if x_new is None:
            if len(history) >= 2:
                # Secant method
                x1, f1 = history[-1]
                x0_s, f0_s = history[-2]
                if abs(f1 - f0_s) > 1e-14:
                    x_new = x1 - f1 * (x1 - x0_s) / (f1 - f0_s)
        
        if x_new is None or not math.isfinite(x_new):
            # Newton with FD derivative
            h_fd = 1e-5
            x_h = x + h_fd
            try:
                f_h = f(x_h)
            except Exception:
                # Try smaller h
                h_fd = 1e-8
                x_h = x + h_fd
                try:
                    f_h = f(x_h)
                except Exception:
                    break
                if not math.isfinite(f_h):
                    break

            if math.isfinite(f_h):
                f_prime = (f_h - fx) / h_fd
                if math.isfinite(f_prime) and abs(f_prime) > 1e-14:
                    x_new = x - fx / f_prime
        
        if x_new is None or not math.isfinite(x_new):
            # Fallback: small step
            x_new = x + 1e-3
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        else:
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                # Shrink step
                x_new = x + 0.5 * (x - x0)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

        # Safeguard: Limit step size
        max_step = max(1.0, abs(x) * 0.5)
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        # Line search / Acceptance
        # Accept if |f_new| < |f|
        if abs(f_new) < abs(fx):
            # Update best
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Move to new point
            # Add to history, keeping last 3 for potential IQI/Secant
            history.append((x_new, f_new))
            if len(history) > 3:
                history.pop(0)
            
            x = x_new
            fx = f_new
        else:
            # Backtrack or try different strategy
            x_try = x + 0.5 * (x_new - x)
            try:
                f_try = f(x_try)
            except Exception:
                break
            if math.isfinite(f_try) and abs(f_try) < abs(fx):
                if abs(f_try) < best_abs_fx:
                    best_x = x_try
                    best_fx = f_try
                    best_abs_fx = abs(f_try)
                if abs(f_try) <= 1e-8:
                    return x_try
                history.append((x_try, f_try))
                if len(history) > 3:
                    history.pop(0)
                x = x_try
                fx = f_try
            else:
                # Stuck. Try a few nearby points to escape
                escaped = False
                for dx in [1e-2, -1e-2, 1e-1, -1e-1, 0.5, -0.5]:
                    x_esc = x + dx
                    try:
                        f_esc = f(x_esc)
                    except Exception:
                        continue
                    if math.isfinite(f_esc) and abs(f_esc) < best_abs_fx:
                        best_x = x_esc
                        best_fx = f_esc
                        best_abs_fx = abs(f_esc)
                        if abs(f_esc) <= 1e-8:
                            return x_esc
                        history.append((x_esc, f_esc))
                        if len(history) > 3:
                            history.pop(0)
                        x = x_esc
                        fx = f_esc
                        escaped = True
                        break
                if not escaped:
                    break

    if abs(best_fx) <= 1e-8:
        return best_x
        
    return best_x
# EVOLVE-BLOCK-END
