"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using adaptive multi-scale scanning
    with caching and robust bisection/secant fallback.
    """
    deadline = time.time() + 145.0
    tol = 1e-10  # tighter than 1e-8 for safety
    
    x = float(x0)
    
    # Cache to avoid redundant function evaluations
    cache = {}
    
    def eval_f(x_val):
        """Cached function evaluation."""
        if x_val in cache:
            return cache[x_val]
        try:
            val = f(x_val)
        except Exception:
            val = float('nan')
        cache[x_val] = val
        return val
    
    best_x = x
    best_f = eval_f(x)
    best_abs = abs(best_f) if best_f is not None and math.isfinite(best_f) else float('inf')
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
        if abs_f <= tol:
            return x_val
        return None
    
    # Check if x0 is already a root
    if best_abs <= tol:
        return best_x
    
    # Adaptive multi-scale scanning strategy
    # Start with fine local search around x0, then expand if needed
    
    # Phase 1: Fine local scan around x0
    base_scale = max(1e-4, abs(x) * 1e-4)
    
    # Try Newton/secant first with a few iterations
    def secant_newton_hybrid(x_start, max_iter=30):
        """Secant method with Newton fallback, returns (x, f_x, converged)."""
        if time.time() > deadline - 5:
            return x_start, eval_f(x_start), False
        
        h = 1e-6 * max(1.0, abs(x_start))
        x0_val = x_start
        x1_val = x_start + h
        f0 = eval_f(x0_val)
        f1 = eval_f(x1_val)
        
        if not math.isfinite(f0) or not math.isfinite(f1):
            return x_start, f0, False
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                return x1_val, f1, False
            
            if abs(f1) <= tol:
                return x1_val, f1, True
            
            if abs(f0) <= tol:
                return x0_val, f0, True
            
            # Secant step
            if f1 != f0:
                x2 = x1 - f1 * (x1 - x0_val) / (f1 - f0)
            else:
                # Fall back to Newton with finite difference
                h_new = 1e-8 * max(1.0, abs(x1_val))
                f_plus = eval_f(x1_val + h_new)
                f_minus = eval_f(x1_val - h_new)
                if math.isfinite(f_plus) and math.isfinite(f_minus):
                    d = (f_plus - f_minus) / (2 * h_new)
                    if d != 0:
                        x2 = x1_val - f1 / d
                    else:
                        break
                else:
                    break
            
            if not math.isfinite(x2):
                break
            
            # Dampen if step is too large
            if abs(x2 - x1_val) > 10 * max(1.0, abs(x1_val)):
                x2 = x1_val + math.copysign(10 * max(1.0, abs(x1_val)), x2 - x1_val)
            
            x0_val = x1_val
            f0 = f1
            x1_val = x2
            f1 = eval_f(x1_val)
            
            if not math.isfinite(f1):
                break
        
        # Return the better of the last few points
        candidates = [(x0_val, f0), (x1_val, f1)]
        if i >= 1:
            candidates.append((x2, eval_f(x2)))
        
        best_local_x = x1_val
        best_local_f = f1
        best_local_abs = abs(f1) if math.isfinite(f1) else float('inf')
        
        for cx, cf in candidates:
            if cf is not None and math.isfinite(cf) and math.isfinite(cx):
                if abs(cf) < best_local_abs:
                    best_local_x = cx
                    best_local_f = cf
                    best_local_abs = abs(cf)
        
        return best_local_x, best_local_f, best_local_abs <= tol
    
    # Try secant/Newton hybrid first
    x_trial, f_trial, converged = secant_newton_hybrid(x)
    result = update_best(x_trial, f_trial)
    if result is not None:
        return result
    
    if converged:
        return x_trial
    
    # Phase 2: Adaptive scanning - fine local, then coarse global
    # Try multiple scales
    scales = []
    for exp in range(-6, 3):
        scale = 10 ** exp * max(1.0, abs(x))
        scales.append(scale)
    
    # Sort by absolute value, fine first
    scales.sort(key=lambda s: abs(s))
    
    found = False
    for scale in scales:
        if time.time() > deadline - 10:
            break
        
        # Scan with multiple points at this scale
        n_points = 20
        for i in range(n_points):
            if time.time() > deadline - 10:
                break
            
            offset = (i / n_points - 0.5) * 2 * scale
            x_scan = x + offset
            
            f_scan = eval_f(x_scan)
            result = update_best(x_scan, f_scan)
            if result is not None:
                return result
            
            # If sign change detected with a nearby point, use bisection
            # Check a few neighbors
            for j in range(i + 1, min(i + 4, n_points)):
                offset_j = (j / n_points - 0.5) * 2 * scale
                x_j = x + offset_j
                f_j = eval_f(x_j)
                
                if f_scan is not None and f_j is not None:
                    if math.isfinite(f_scan) and math.isfinite(f_j):
                        if f_scan * f_j < 0:
                            # Sign change - use bisection
                            a, b = x_scan, x_j
                            fa, fb = f_scan, f_j
                            
                            for _ in range(60):
                                if time.time() > deadline - 5:
                                    break
                                mid = (a + b) / 2
                                fm = eval_f(mid)
                                if not math.isfinite(fm):
                                    break
                                if abs(fm) <= tol:
                                    return mid
                                if fa * fm <= 0:
                                    b = mid
                                    fb = fm
                                else:
                                    a = mid
                                    fa = fm
                            
                            # Return best from bisection
                            result = update_best(a, fa)
                            result2 = update_best(b, fb)
                            if best_abs <= tol:
                                return best_x
        
        # If we found something good at this scale, stop expanding
        if best_abs < 1e-6:
            break
    
    # Phase 3: Broader scan if still not found
    if best_abs > 1e-6 and time.time() < deadline - 20:
        # Try larger scales
        for exp in range(3, 6):
            if time.time() > deadline - 10:
                break
            scale = 10 ** exp * max(1.0, abs(x))
            n_points = 10
            for i in range(n_points):
                if time.time() > deadline - 5:
                    break
                offset = (i / n_points - 0.5) * 2 * scale
                x_scan = x + offset
                f_scan = eval_f(x_scan)
                result = update_best(x_scan, f_scan)
                if result is not None:
                    return result
    
    # Final: return best found
    if best_abs <= 1e-8:
        return best_x
    
    # If we haven't found a good root, do one more secant attempt from best_x
    if time.time() < deadline - 5:
        x_trial, f_trial, converged = secant_newton_hybrid(best_x, max_iter=20)
        update_best(x_trial, f_trial)
    
    return best_x
# EVOLVE-BLOCK-END
