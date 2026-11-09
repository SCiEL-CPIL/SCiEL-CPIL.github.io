"""Model-and-optimize root finder for ill-conditioned equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 118.0  # safety margin before 120s
    _MAX_EVALS = 480
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            return float(f(x))
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _budget_ok():
        return _evals < _MAX_EVALS

    def _safe(x):
        return math.isfinite(x)

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if _safe(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not _safe(fx):
            return False
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)
            return af <= 1e-8
        return False

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # Inverse quadratic interpolation (IQI) - fits parabola through 3 points
    def _iqi(x1, f1, x2, f2, x3, f3):
        """
        Inverse quadratic interpolation: fit parabola x = q(f) through
        three points (f1,x1), (f2,x2), (f3,x3) and find x where q(f)=0.
        """
        if not all(_safe(v) for v in [x1, f1, x2, f2, x3, f3]):
            return None
        # Lagrange interpolation for x as function of f
        denom = (f1 - f2) * (f1 - f3) * (f2 - f3)
        if abs(denom) < 1e-30:
            return None
        x0_iqi = (f2 * f3 * x1 / ((f1 - f2) * (f1 - f3)) +
                  f1 * f3 * x2 / ((f2 - f1) * (f2 - f3)) +
                  f1 * f2 * x3 / ((f3 - f1) * (f3 - f2)))
        if not _safe(x0_iqi):
            return None
        return x0_iqi

    # Secant method step
    def _secant(x1, f1, x2, f2):
        """Secant method: linear interpolation between two points."""
        if not all(_safe(v) for v in [x1, f1, x2, f2]):
            return None
        denom = f2 - f1
        if abs(denom) < 1e-30:
            return None
        x_new = x1 - f1 * (x2 - x1) / denom
        if not _safe(x_new):
            return None
        return x_new

    # Bisection
    def _bisect(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if not _time_ok() or not _budget_ok():
                break
            mid = (a + b) / 2.0
            fmid = _f(mid)
            if not _safe(fmid):
                break
            _update(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return (a + b) / 2.0

    # --- Phase 1: IQI/Secant with bracket tracking ---
    # Initialize with two points for secant
    x1 = float(x0)
    f1 = _f(x1)
    if not _safe(f1):
        x1 = best_x
        f1 = best_fx
    _update(x1, f1)
    
    # Get second point for secant
    scale = max(1.0, abs(x1))
    x2 = x1 + 0.1 * scale
    f2 = _f(x2)
    _update(x2, f2)
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2
    
    # Keep track of last 3 points for IQI
    points = [(x1, f1), (x2, f2)]
    
    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None
    
    for i in range(60):
        if not _time_ok() or not _budget_ok():
            break
        
        x_prev = points[-1][0]
        f_prev = points[-1][1]
        
        if abs(f_prev) <= 1e-8:
            return x_prev
        
        # Determine next step
        x_new = None
        
        if len(points) >= 3:
            # Try IQI first
            x3, f3 = points[-3]
            x2_, f2_ = points[-2]
            x1_, f1_ = points[-1]
            x_new = _iqi(x1_, f1_, x2_, f2_, x3, f3)
        
        if x_new is None and len(points) >= 2:
            # Fallback to secant
            x_a, f_a = points[-2]
            x_b, f_b = points[-1]
            x_new = _secant(x_a, f_a, x_b, f_b)
        
        if x_new is None:
            # Fallback: step toward best or small perturbation
            x_new = x_prev + 0.1 * max(1.0, abs(x_prev))
        
        # Safeguard: don't go too far
        max_step = 5.0 * max(1.0, abs(x_prev))
        if abs(x_new - x_prev) > max_step:
            x_new = x_prev + max_step * (1 if x_new > x_prev else -1)
        
        if not _safe(x_new):
            x_new = x_prev + 0.01 * max(1.0, abs(x_prev))
            if not _safe(x_new):
                break
        
        # Check if we're repeating a point
        if any(abs(x_new - px) < 1e-15 * max(1.0, abs(px)) for px, _ in points):
            x_new = x_new + 1e-10 * max(1.0, abs(x_new))
        
        f_new = _f(x_new)
        if not _safe(f_new):
            # Try a smaller step
            x_new = x_prev - 0.5 * (x_new - x_prev)
            f_new = _f(x_new)
            if not _safe(f_new):
                break
        
        _update(x_new, f_new)
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Track bracket
        if (f_new <= 0 and f_prev >= 0) or (f_new >= 0 and f_prev <= 0):
            bracket_a, bracket_b = x_prev, x_new
            bracket_fa, bracket_fb = f_prev, f_new
            if abs(bracket_fa) > abs(bracket_fb):
                bracket_a, bracket_b = bracket_b, bracket_a
                bracket_fa, bracket_fb = bracket_fb, bracket_fa
        
        # Update points history (keep last 3)
        points.append((x_new, f_new))
        if len(points) > 3:
            points.pop(0)
        
        # Convergence check
        if abs(x_new - x_prev) < 1e-15 * max(1.0, abs(x_new)):
            break

    # --- Phase 2: If we have a bracket, use bisection with IQI acceleration ---
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            a, b = bracket_a, bracket_b
            fa, fb = bracket_fa, bracket_fb
            # Keep a history for IQI within bracket
            hist = [(a, fa), (b, fb)]
            
            for i in range(40):
                if not _time_ok() or not _budget_ok():
                    break
                
                # Try IQI if we have 3 points
                x_trial = None
                if len(hist) >= 3:
                    x_trial = _iqi(hist[0][0], hist[0][1], 
                                   hist[1][0], hist[1][1],
                                   hist[2][0], hist[2][1])
                
                # Fallback to bisection midpoint
                if x_trial is None or not _safe(x_trial) or not (a < x_trial < b):
                    x_trial = (a + b) / 2.0
                    if not _safe(x_trial):
                        break
                
                f_trial = _f(x_trial)
                if not _safe(f_trial):
                    break
                _update(x_trial, f_trial)
                if abs(f_trial) <= 1e-8:
                    return x_trial
                
                # Update bracket
                if (f_trial <= 0 and fa <= 0) or (f_trial >= 0 and fa >= 0):
                    a, fa = x_trial, f_trial
                else:
                    b, fb = x_trial, f_trial
                
                # Update history
                hist.append((x_trial, f_trial))
                if len(hist) > 3:
                    hist.pop(0)
                
                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break
            
            res = (a + b) / 2.0
            fres = _f(res)
            _update(res, fres)
            if abs(fres) <= 1e-8:
                return res

    # --- Phase 3: Scan for brackets ---
    if not _time_ok() or not _budget_ok():
        return best_x

    base = best_x
    scan_scale = max(1.0, abs(base))
    found = False
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        if not _time_ok() or not _budget_ok():
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _f(a)
            fb = _f(b)
            if not (_safe(fa) and _safe(fb)):
                continue
            _update(a, fa)
            _update(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisect(a, b, fa, fb, max_iter=50)
                fres = _f(res)
                _update(res, fres)
                if abs(fres) <= 1e-8:
                    return res
                found = True
                break
        if found:
            break

    # --- Phase 4: Minimize |f| via golden section ---
    if not _time_ok() or not _budget_ok():
        return best_x

    center = best_x
    scale = max(1.0, abs(center))
    a = center - 20.0 * scale
    b = center + 20.0 * scale

    # Ensure finite values at boundaries
    fa = _f(a)
    fb = _f(b)
    if not (_safe(fa) and _safe(fb)):
        for mult in [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
            a = center - mult * scale
            b = center + mult * scale
            fa = _f(a)
            fb = _f(b)
            if _safe(fa) and _safe(fb):
                break
            a, b = center, center
            break

    if _safe(fa) and _safe(fb) and a != b:
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _f(c)
        fd = _f(d)
        _update(c, fc)
        _update(d, fd)

        for _ in range(40):
            if not _time_ok() or not _budget_ok():
                break
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fc) < abs(fd):
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = _f(c)
                _update(c, fc)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = _f(d)
                _update(d, fd)
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        fmid = _f(mid)
        _update(mid, fmid)
        if abs(fmid) <= 1e-8:
            return mid

    # --- Phase 5: Final secant polish with multiple strategies ---
    if _time_ok() and _budget_ok():
        # Try multiple starting points for secant polish
        candidates = [best_x]
        
        # Add nearby points if we have them
        if len(points) >= 2:
            candidates.extend([p[0] for p in points[-3:]])
        
        # Try each candidate as a starting point for secant
        for start_x in candidates:
            if not _time_ok() or not _budget_ok():
                break
            
            x_a = start_x
            f_a = _f(x_a)
            if not _safe(f_a):
                continue
            
            _update(x_a, f_a)
            if abs(f_a) <= 1e-8:
                return x_a
            
            # Find a second point - try several offsets
            scale = max(1.0, abs(x_a))
            second_found = False
            for offset_factor in [0.01, 0.001, 0.1, 0.0001, 0.01]:
                for sign in [1, -1]:
                    if not _time_ok() or not _budget_ok():
                        break
                    x_b = x_a + sign * offset_factor * scale
                    if x_b == x_a:
                        continue
                    f_b = _f(x_b)
                    if _safe(f_b):
                        _update(x_b, f_b)
                        if abs(f_b) <= 1e-8:
                            return x_b
                        second_found = True
                        break
                if second_found:
                    break
            
            if not second_found:
                continue
            
            # Secant iterations
            for _ in range(20):
                if not _time_ok() or not _budget_ok():
                    break
                if abs(f_a) <= 1e-8:
                    return x_a
                if abs(f_b) <= 1e-8:
                    return x_b
                
                x_new = _secant(x_a, f_a, x_b, f_b)
                if x_new is None:
                    break
                
                # Safeguard
                max_step = 5.0 * max(1.0, abs(x_a))
                if abs(x_new - x_a) > max_step:
                    x_new = x_a + max_step * (1 if x_new > x_a else -1)
                
                if not _safe(x_new):
                    break
                
                f_new = _f(x_new)
                if not _safe(f_new):
                    break
                
                _update(x_new, f_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                
                # Keep the two points with smaller |f|
                if abs(f_new) < abs(f_a):
                    x_a, f_a = x_b, f_b
                    x_b, f_b = x_new, f_new
                else:
                    x_a, f_a = x_b, f_b
                    x_b, f_b = x_new, f_new
                
                # Check for convergence
                if abs(x_new - x_a) < 1e-14 * max(1.0, abs(x_new)):
                    break
    
    # --- Phase 6: Very fine local search around best_x ---
    if _time_ok() and _budget_ok():
        center = best_x
        scale = max(1.0, abs(center))
        
        # Try a grid of small perturbations
        for factor in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            if not _time_ok() or not _budget_ok():
                break
            for sign in [1, -1]:
                x_try = center + sign * factor * scale
                if x_try == center:
                    continue
                f_try = _f(x_try)
                if _safe(f_try):
                    _update(x_try, f_try)
                    if abs(f_try) <= 1e-8:
                        return x_try
    
    return best_x
# EVOLVE-BLOCK-END
