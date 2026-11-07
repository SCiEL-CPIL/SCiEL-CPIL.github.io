"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using spectral-aware bracketing:
    1. Estimate dominant period via frequency sweep
    2. Place dense brackets using estimated period
    3. Bisection + Newton hybrid for convergence
    """
    x0 = float(x0)
    calls = 0
    
    def _f(x):
        nonlocal calls
        calls += 1
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    
    # Step 1: Check if x0 is already a root
    fx0 = _f(x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0
    
    # Step 2: Spectral-aware period estimation
    # Sample at points spaced by 1/(2*pi*k) for various k to identify dominant period
    # For oscillatory functions, we expect sign changes at regular intervals
    
    # First, try a quick local Newton step
    x = x0
    fx = fx0
    for _ in range(5):
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-7 * max(1.0, abs(x))
        fph = _f(x + h)
        if not math.isfinite(fph):
            break
        d = (fph - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = _f(x)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x
    
    # Step 3: Estimate period via adaptive frequency sweep
    # Try different k values: spacing = 1/(2*pi*k)
    # We'll sample a few points at each spacing and look for sign changes
    
    best_period = None
    best_bracket = None
    
    # Try a range of k values (frequencies)
    # k=1 gives spacing ~0.159, k=2 gives ~0.079, etc.
    # Also try larger spacings for slower oscillations
    
    # Strategy: for each candidate period, sample a few points around x0
    # and check for sign changes
    
    candidate_periods = []
    
    # Generate candidate periods based on common oscillatory frequencies
    # Assume period T = 2*pi/k for various k
    for k in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]:
        T = 2 * math.pi / k
        candidate_periods.append(T)
    
    # Also try some empirical periods
    candidate_periods.extend([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    
    # Sort by period (smallest first)
    candidate_periods.sort()
    
    for T in candidate_periods:
        if T < 1e-6:
            continue
        if T > 1000:
            continue
        
        # Sample points around x0 with spacing T/2 (half period)
        # Check for sign changes in a window of ~2 periods
        half_T = T / 2.0
        n_samples = 10  # Sample 10 points in each direction
        
        prev_x = None
        prev_f = None
        
        # Sweep from x0 - 2*T to x0 + 2*T
        for i in range(-n_samples, n_samples + 1):
            xi = x0 + i * half_T
            fi = _f(xi)
            if not math.isfinite(fi):
                prev_x = None
                prev_f = None
                continue
            if abs(fi) <= 1e-8:
                return xi
            if prev_f is not None and math.isfinite(prev_f) and prev_f * fi < 0:
                # Found a bracket!
                a, b = prev_x, xi
                fa, fb = prev_f, fi
                best_period = T
                best_bracket = (a, b, fa, fb)
                break
            prev_x = xi
            prev_f = fi
        
        if best_bracket is not None:
            break
    
    # If no bracket found via spectral method, fall back to geometric expansion
    if best_bracket is None:
        # Try geometric expansion in both directions
        step = max(1e-3, abs(x0) * 1e-3)
        for direction in [1, -1]:
            for i in range(20):
                xi = x0 + direction * step * (2 ** i)
                fi = _f(xi)
                if not math.isfinite(fi):
                    continue
                if abs(fi) <= 1e-8:
                    return xi
                if math.isfinite(fx0) and fx0 * fi < 0:
                    if direction == 1:
                        a, b = x0, xi
                        fa, fb = fx0, fi
                    else:
                        a, b = xi, x0
                        fa, fb = fi, fx0
                    best_bracket = (a, b, fa, fb)
                    break
            if best_bracket is not None:
                break
    
    # Final fallback: linear sweep
    if best_bracket is None:
        base_step = max(1e-4, abs(x0) * 1e-4)
        prev_x = None
        prev_f = None
        for i in range(-50, 51):
            xi = x0 + i * base_step
            fi = _f(xi)
            if not math.isfinite(fi):
                prev_x = None
                prev_f = None
                continue
            if abs(fi) <= 1e-8:
                return xi
            if prev_f is not None and math.isfinite(prev_f) and prev_f * fi < 0:
                a, b = prev_x, xi
                fa, fb = prev_f, fi
                best_bracket = (a, b, fa, fb)
                break
            prev_x = xi
            prev_f = fi
    
    # If still no bracket, return best Newton estimate
    if best_bracket is None:
        x = x0
        fx = _f(x)
        for _ in range(50):
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fph = _f(x + h)
            if not math.isfinite(fph):
                break
            d = (fph - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = _f(x)
        return x if math.isfinite(x) else x0
    
    # Step 4: Bisection + Newton hybrid to converge within the bracket
    a, b, fa, fb = best_bracket
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    lo, hi = a, b
    flo, fhi = fa, fb
    
    for _ in range(200):
        if not math.isfinite(flo) or not math.isfinite(fhi):
            break
        if abs(flo) <= 1e-8:
            return lo
        if abs(fhi) <= 1e-8:
            return hi
        
        mid = (lo + hi) / 2.0
        fmid = _f(mid)
        if not math.isfinite(fmid):
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            continue
        
        if abs(fmid) <= 1e-8:
            return mid
        
        # Try Newton step
        h = 1e-8 * max(1.0, abs(mid))
        fmid_h = _f(mid + h)
        if math.isfinite(fmid_h):
            d = (fmid_h - fmid) / h
            if d != 0 and math.isfinite(d):
                x_new = mid - fmid / d
                if math.isfinite(x_new) and lo < x_new < hi:
                    fnew = _f(x_new)
                    if math.isfinite(fnew):
                        if abs(fnew) <= 1e-8:
                            return x_new
                        if flo * fnew < 0:
                            hi = x_new
                            fhi = fnew
                        else:
                            lo = x_new
                            flo = fnew
                        continue
        
        # Bisection step
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Return the best estimate
    candidates = [lo, hi, (lo + hi) / 2.0]
    best_x = lo
    best_f = abs(flo) if math.isfinite(flo) else float('inf')
    for c in candidates:
        if not math.isfinite(c):
            continue
        fc = _f(c)
        if math.isfinite(fc) and abs(fc) < best_f:
            best_f = abs(fc)
            best_x = c
    
    return best_x
# EVOLVE-BLOCK-END
