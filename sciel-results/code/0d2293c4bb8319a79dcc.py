"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    MAX_CALLS = 450
    DEADLINE = 110.0  # seconds, safety margin before 120s limit
    start_time = time.time()
    
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        if time.time() - start_time > DEADLINE:
            return None
        calls[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return None
        return val
    
    def elapsed():
        return time.time() - start_time
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # Store points for interpolation: list of (x, f(x))
    points = [(x, fx)]
    
    def update_best(xv, fv):
        nonlocal best_x, best_fx, best_abs
        if fv is not None and math.isfinite(fv):
            av = abs(fv)
            if av < best_abs:
                best_x = xv
                best_fx = fv
                best_abs = av
    
    def check_done(xv, fv):
        if fv is not None and abs(fv) <= 1e-8:
            return True
        return False
    
    # Phase 1: Try to find a bracket around x0
    step = 1e-2 * max(1.0, abs(x))
    a = x - step
    b = x + step
    
    fa = feval(a)
    fb = feval(b)
    
    if fa is not None:
        points.append((a, fa))
        update_best(a, fa)
        if check_done(a, fa):
            return a
    if fb is not None:
        points.append((b, fb))
        update_best(b, fb)
        if check_done(b, fb):
            return b
    
    # Check if we have a bracket
    has_bracket = False
    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None
    
    if fa is not None and fb is not None and fa * fb < 0:
        has_bracket = True
        bracket_a, bracket_b = a, b
        bracket_fa, bracket_fb = fa, fb
    elif fa is not None and fx * fa < 0:
        has_bracket = True
        bracket_a, bracket_b = a, x
        bracket_fa, bracket_fb = fa, fx
    elif fb is not None and fx * fb < 0:
        has_bracket = True
        bracket_a, bracket_b = x, b
        bracket_fa, bracket_fb = fx, fb
    
    # Try expanding bracket if not found
    if not has_bracket:
        for mult in [0.1, 1.0, 10.0, 100.0]:
            if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
                break
            step2 = mult * max(1.0, abs(x))
            a2 = x - step2
            b2 = x + step2
            fa2 = feval(a2)
            fb2 = feval(b2)
            if fa2 is not None:
                points.append((a2, fa2))
                update_best(a2, fa2)
                if check_done(a2, fa2):
                    return a2
            if fb2 is not None:
                points.append((b2, fb2))
                update_best(b2, fb2)
                if check_done(b2, fb2):
                    return b2
            
            # Check all pairs for a sign change
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    xi, fi = points[i]
                    xj, fj = points[j]
                    if fi * fj < 0:
                        has_bracket = True
                        if abs(xi - x) <= abs(xj - x):
                            bracket_a, bracket_b = xi, xj
                        else:
                            bracket_a, bracket_b = xj, xi
                        bracket_fa, bracket_fb = fi, fj
                        break
                if has_bracket:
                    break
            if has_bracket:
                break
    
    # Phase 2: Main iteration loop
    # Use a combination of methods based on available information
    last_x = x
    last_fx = fx
    
    for iteration in range(200):
        if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
            break
        
        # Try inverse quadratic interpolation if we have 3+ points with distinct x values
        # Filter points to unique x values
        unique_points = []
        seen_x = set()
        for px, pv in points:
            key = round(px, 12)
            if key not in seen_x:
                seen_x.add(key)
                unique_points.append((px, pv))
        
        if len(unique_points) >= 3:
            # Use the 3 most recent points for IQUAD
            recent = unique_points[-3:]
            x1, f1 = recent[0]
            x2, f2 = recent[1]
            x3, f3 = recent[2]
            
            # Inverse quadratic interpolation: fit quadratic through (f1,x1),(f2,x2),(f3,x3)
            # Find f=0
            try:
                # Lagrange interpolation for x as function of f
                if f1 != f2 and f2 != f3 and f1 != f3:
                    # x(f) = x1 * L1(f) + x2 * L2(f) + x3 * L3(f)
                    # L1(f) = (f-f2)(f-f3) / ((f1-f2)(f1-f3))
                    # L2(f) = (f-f1)(f-f3) / ((f2-f1)(f2-f3))
                    # L3(f) = (f-f1)(f-f2) / ((f3-f1)(f3-f2))
                    # At f=0:
                    # x(0) = x1 * (f2*f3)/((f1-f2)(f1-f3)) + x2 * (f1*f3)/((f2-f1)(f2-f3)) + x3 * (f1*f2)/((f3-f1)(f3-f2))
                    denom1 = (f1 - f2) * (f1 - f3)
                    denom2 = (f2 - f1) * (f2 - f3)
                    denom3 = (f3 - f1) * (f3 - f2)
                    
                    if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                        x_iquad = (x1 * (f2 * f3) / denom1 + 
                                   x2 * (f1 * f3) / denom2 + 
                                   x3 * (f1 * f2) / denom3)
                        
                        if math.isfinite(x_iquad):
                            # Check if x_iquad is within a reasonable range
                            if has_bracket and bracket_a is not None and bracket_b is not None:
                                lo = min(bracket_a, bracket_b)
                                hi = max(bracket_a, bracket_b)
                                if lo - 1e-10 <= x_iquad <= hi + 1e-10:
                                    fx_new = feval(x_iquad)
                                    if fx_new is not None:
                                        points.append((x_iquad, fx_new))
                                        update_best(x_iquad, fx_new)
                                        if check_done(x_iquad, fx_new):
                                            return x_iquad
                                        last_x = x_iquad
                                        last_fx = fx_new
                                        continue
                            else:
                                # No bracket, accept if not too far
                                if abs(x_iquad - last_x) < 10 * max(1.0, abs(last_x)):
                                    fx_new = feval(x_iquad)
                                    if fx_new is not None:
                                        points.append((x_iquad, fx_new))
                                        update_best(x_iquad, fx_new)
                                        if check_done(x_iquad, fx_new):
                                            return x_iquad
                                        last_x = x_iquad
                                        last_fx = fx_new
                                        continue
            except (ZeroDivisionError, OverflowError):
                pass
        
        # Try cubic interpolation if we have 4+ points
        if len(unique_points) >= 4 and calls[0] < MAX_CALLS - 10:
            recent4 = unique_points[-4:]
            try:
                x_vals = np.array([p[0] for p in recent4])
                f_vals = np.array([p[1] for p in recent4])
                # Fit cubic polynomial
                coeffs = np.polyfit(x_vals, f_vals, 3)
                # Find roots of the cubic
                roots = np.roots(coeffs)
                # Filter real roots
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                
                # Pick the root closest to last_x or within bracket
                best_root = None
                best_dist = float('inf')
                for rr in real_roots:
                    if math.isfinite(rr):
                        if has_bracket and bracket_a is not None and bracket_b is not None:
                            lo = min(bracket_a, bracket_b)
                            hi = max(bracket_a, bracket_b)
                            if lo <= rr <= hi:
                                best_root = rr
                                break
                        else:
                            dist = abs(rr - last_x)
                            if dist < best_dist:
                                best_dist = dist
                                best_root = rr
                
                if best_root is not None and math.isfinite(best_root):
                    fx_new = feval(best_root)
                    if fx_new is not None:
                        points.append((best_root, fx_new))
                        update_best(best_root, fx_new)
                        if check_done(best_root, fx_new):
                            return best_root
                        last_x = best_root
                        last_fx = fx_new
                        continue
            except Exception:
                pass
        
        # Secant method if we have 2 points
        if len(unique_points) >= 2:
            x1, f1 = unique_points[-2]
            x2, f2 = unique_points[-1]
            if abs(f2 - f1) > 1e-30:
                x_secant = x2 - f2 * (x2 - x1) / (f2 - f1)
                if math.isfinite(x_secant):
                    if has_bracket and bracket_a is not None and bracket_b is not None:
                        lo = min(bracket_a, bracket_b)
                        hi = max(bracket_a, bracket_b)
                        if lo - 1e-10 <= x_secant <= hi + 1e-10:
                            fx_new = feval(x_secant)
                            if fx_new is not None:
                                points.append((x_secant, fx_new))
                                update_best(x_secant, fx_new)
                                if check_done(x_secant, fx_new):
                                    return x_secant
                                last_x = x_secant
                                last_fx = fx_new
                                continue
                    else:
                        if abs(x_secant - last_x) < 100 * max(1.0, abs(last_x)):
                            fx_new = feval(x_secant)
                            if fx_new is not None:
                                points.append((x_secant, fx_new))
                                update_best(x_secant, fx_new)
                                if check_done(x_secant, fx_new):
                                    return x_secant
                                last_x = x_secant
                                last_fx = fx_new
                                continue
        
        # Bisection if we have a bracket
        if has_bracket and bracket_a is not None and bracket_b is not None:
            mid = (bracket_a + bracket_b) / 2.0
            if abs(mid - bracket_a) < 1e-15 * max(1.0, abs(bracket_a)):
                break
            fmid = feval(mid)
            if fmid is not None:
                points.append((mid, fmid))
                update_best(mid, fmid)
                if check_done(mid, fmid):
                    return mid
                
                # Update bracket
                if bracket_fa * fmid < 0:
                    bracket_b = mid
                    bracket_fb = fmid
                else:
                    bracket_a = mid
                    bracket_fa = fmid
                last_x = mid
                last_fx = fmid
                continue
        
        # No bracket, no good interpolation - try stepping
        if len(points) < 2:
            # Shouldn't happen, but safeguard
            break
        
        # Try a few different step directions
        stepped = False
        for direction in [1.0, -1.0]:
            if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
                break
            step_size = 0.1 * max(1.0, abs(last_x))
            x_new = last_x + direction * step_size
            fx_new = feval(x_new)
            if fx_new is not None:
                points.append((x_new, fx_new))
                update_best(x_new, fx_new)
                if check_done(x_new, fx_new):
                    return x_new
                last_x = x_new
                last_fx = fx_new
                stepped = True
                break
        
        if not stepped:
            # Try smaller steps
            for mult in [0.01, 0.001]:
                if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
                    break
                step_size = mult * max(1.0, abs(last_x))
                for direction in [1.0, -1.0]:
                    if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
                        break
                    x_new = last_x + direction * step_size
                    fx_new = feval(x_new)
                    if fx_new is not None:
                        points.append((x_new, fx_new))
                        update_best(x_new, fx_new)
                        if check_done(x_new, fx_new):
                            return x_new
                        last_x = x_new
                        last_fx = fx_new
                        break
                if calls[0] >= MAX_CALLS or elapsed() > DEADLINE:
                    break
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
