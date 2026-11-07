"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
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

    # History for interpolation: list of (x, fx), keep last 3
    history = [(x, fx)]
    
    # Parameters
    max_iter = 30  # Reduced since quadratic interpolation converges faster
    h_fd = 1e-5  # Finite difference step
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        x_new = None
        
        # Try quadratic interpolation if we have 3+ distinct points
        if len(history) >= 3:
            x0_h, f0_h = history[-3]
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            
            # Check points are sufficiently distinct
            if (abs(x1_h - x0_h) > 1e-12 and abs(x2_h - x1_h) > 1e-12 and 
                abs(x2_h - x0_h) > 1e-12):
                # Fit quadratic: f(x) = a(x-x0)(x-x1)(x-x2) + interpolation
                # Use Lagrange interpolation to find root
                # Solve for x where quadratic = 0
                # Using formula for root of quadratic through 3 points
                
                # Compute Lagrange coefficients
                denom0 = (x0_h - x1_h) * (x0_h - x2_h)
                denom1 = (x1_h - x0_h) * (x1_h - x2_h)
                denom2 = (x2_h - x0_h) * (x2_h - x1_h)
                
                if abs(denom0) > 1e-14 and abs(denom1) > 1e-14 and abs(denom2) > 1e-14:
                    # Coefficients of quadratic: A*x^2 + B*x + C
                    A = f0_h/denom0 + f1_h/denom1 + f2_h/denom2
                    B = -f0_h*(x1_h+x2_h)/denom0 - f1_h*(x0_h+x2_h)/denom1 - f2_h*(x0_h+x1_h)/denom2
                    C = f0_h*x1_h*x2_h/denom0 + f1_h*x0_h*x2_h/denom1 + f2_h*x0_h*x1_h/denom2
                    
                    if abs(A) > 1e-14:
                        disc = B*B - 4*A*C
                        if disc >= 0:
                            sqrt_disc = math.sqrt(disc)
                            # Two roots; pick the one closer to current x or with smaller |f|
                            r1 = (-B + sqrt_disc) / (2*A)
                            r2 = (-B - sqrt_disc) / (2*A)
                            
                            # Choose root closer to current x
                            if abs(r1 - x) < abs(r2 - x):
                                x_new = r1
                            else:
                                x_new = r2
                        else:
                            # Complex roots, use secant
                            x_new = None
                    else:
                        # Linear case
                        if abs(B) > 1e-14:
                            x_new = -C / B
                        else:
                            x_new = None
        
        # Fallback to secant if quadratic failed or we have < 3 points
        if x_new is None or not math.isfinite(x_new):
            if len(history) >= 2:
                x1_h, f1_h = history[-1]
                x0_h, f0_h = history[-2]
                if abs(f1_h - f0_h) > 1e-14 and abs(x1_h - x0_h) > 1e-14:
                    x_new = x1_h - f1_h * (x1_h - x0_h) / (f1_h - f0_h)
        
        # Fallback to Newton with FD if secant failed
        if x_new is None or not math.isfinite(x_new):
            x_h = x + h_fd
            try:
                f_h = f(x_h)
            except Exception:
                x_h = x + 1e-8
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
        
        # Final fallback: small step
        if x_new is None or not math.isfinite(x_new):
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

        # Acceptance: require |f_new| < |f|
        if abs(f_new) < abs(fx):
            # Update best
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Move to new point
            history.append((x_new, f_new))
            if len(history) > 3:
                history.pop(0)
            
            x = x_new
            fx = f_new
        else:
            # Backtrack: try half step
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
                # Stuck - try a few perturbations
                for dx in [1e-4, -1e-4, 1e-3, -1e-3]:
                    x_pert = x + dx
                    try:
                        f_pert = f(x_pert)
                    except Exception:
                        continue
                    if math.isfinite(f_pert) and abs(f_pert) < abs(fx):
                        if abs(f_pert) < best_abs_fx:
                            best_x = x_pert
                            best_fx = f_pert
                            best_abs_fx = abs(f_pert)
                        if abs(f_pert) <= 1e-8:
                            return x_pert
                        history.append((x_pert, f_pert))
                        if len(history) > 3:
                            history.pop(0)
                        x = x_pert
                        fx = f_pert
                        break
                else:
                    break

    # Final polish pass: if we're close, do Newton iterations with tight tolerance
    if abs(best_fx) <= 1e-4:
        x_polish = best_x
        fx_polish = best_fx
        for _ in range(10):
            if time.time() > deadline:
                break
            h = 1e-7
            try:
                f_h = f(x_polish + h)
            except Exception:
                break
            if not math.isfinite(f_h):
                break
            fp = (f_h - fx_polish) / h
            if not math.isfinite(fp) or abs(fp) < 1e-14:
                break
            x_new = x_polish - fx_polish / fp
            if not math.isfinite(x_new):
                break
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < abs(fx_polish):
                x_polish = x_new
                fx_polish = f_new
            else:
                break
        if abs(fx_polish) < abs(best_fx):
            best_x = x_polish
            best_fx = fx_polish
            best_abs_fx = abs(fx_polish)

    if abs(best_fx) <= 1e-8:
        return best_x
        
    return best_x
# EVOLVE-BLOCK-END
