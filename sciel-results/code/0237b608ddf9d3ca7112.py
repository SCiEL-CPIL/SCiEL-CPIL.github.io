import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Hybrid Secant/Bisection.
    """
    x0 = float(x0)
    tol = 1e-8
    max_calls = 450
    call_count = 0
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        call_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            return val
        except:
            return None

    # Phase 1: Secant with Bracket Detection
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0
    x2 = x0 + h
    
    f1 = safe_f(x1)
    if f1 is None:
        return x1
    if abs(f1) <= tol:
        return x1
        
    f2 = safe_f(x2)
    if f2 is None:
        return x1

    if abs(f2) <= tol:
        return x2

    best_x = x1 if abs(f1) < abs(f2) else x2
    best_f = min(abs(f1), abs(f2))
    
    # Check for initial bracket
    if f1 * f2 < 0:
        left, right = x1, x2
        f_left, f_right = f1, f2
        # Bisection
        for _ in range(100):
            if call_count >= max_calls:
                break
            mid = (left + right) / 2.0
            f_mid = safe_f(mid)
            if f_mid is None:
                break
            if abs(f_mid) <= tol:
                return mid
            if f_left * f_mid <= 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
            if abs(right - left) < 1e-12:
                break
        # Return best of the last few
        candidates = [left, right]
        if 'f_mid' in locals() and f_mid is not None:
            candidates.append(mid)
        # Compare
        best_ret = left
        min_val = abs(f_left)
        if abs(f_right) < min_val:
            best_ret = right
            min_val = abs(f_right)
        if 'f_mid' in locals() and f_mid is not None and abs(f_mid) < min_val:
            best_ret = mid
        return best_ret

    for _ in range(40):
        if call_count >= max_calls:
            break
            
        denom = f2 - f1
        if abs(denom) < 1e-15:
            break
            
        x3 = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x3):
            break
            
        f3 = safe_f(x3)
        if f3 is None:
            break
            
        if abs(f3) <= tol:
            return x3
            
        if abs(f3) < best_f:
            best_f = abs(f3)
            best_x = x3
            
        # Check for bracket with new point
        if f2 * f3 < 0:
            # Bracket found between x2 and x3
            if abs(f2) < abs(f3):
                left, right = x2, x3
                f_left, f_right = f2, f3
            else:
                left, right = x3, x2
                f_left, f_right = f3, f2
            # Bisection
            for _ in range(100):
                if call_count >= max_calls:
                    break
                mid = (left + right) / 2.0
                f_mid = safe_f(mid)
                if f_mid is None:
                    break
                if abs(f_mid) <= tol:
                    return mid
                if f_left * f_mid <= 0:
                    right = mid
                    f_right = f_mid
                else:
                    left = mid
                    f_left = f_mid
                if abs(right - left) < 1e-12:
                    break
            # Return best of the last few
            candidates = [left, right]
            if 'f_mid' in locals() and f_mid is not None:
                candidates.append(mid)
            best_ret = left
            min_val = abs(f_left)
            if abs(f_right) < min_val:
                best_ret = right
                min_val = abs(f_right)
            if 'f_mid' in locals() and f_mid is not None and abs(f_mid) < min_val:
                best_ret = mid
            return best_ret

        # Check for divergence
        if abs(f3) > abs(f2) and abs(f3) > abs(f1):
            break
            
        x1, f1 = x2, f2
        x2, f2 = x3, f3

    # Phase 2: If no bracket found and secant didn't converge
    # Try to bracket around best_x
    left = best_x - 1.0
    right = best_x + 1.0
    
    f_left = safe_f(left)
    f_right = safe_f(right)
    
    if f_left is None or f_right is None:
        return best_x
        
    for _ in range(20):
        if call_count >= max_calls:
            break
            
        if f_left * f_right < 0:
            break
            
        # Expand
        width = right - best_x
        left = best_x - width * 2
        right = best_x + width * 2
        
        f_left = safe_f(left)
        f_right = safe_f(right)
        
        if f_left is None or f_right is None:
            return best_x

    if f_left * f_right < 0:
        # Bisection
        for _ in range(100):
            if call_count >= max_calls:
                break
            mid = (left + right) / 2.0
            f_mid = safe_f(mid)
            if f_mid is None:
                break
            if abs(f_mid) <= tol:
                return mid
            if f_left * f_mid <= 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
            if abs(right - left) < 1e-12:
                break
        # Return best of the last few
        candidates = [left, right]
        if 'f_mid' in locals() and f_mid is not None:
            candidates.append(mid)
        best_ret = left
        min_val = abs(f_left)
        if abs(f_right) < min_val:
            best_ret = right
            min_val = abs(f_right)
        if 'f_mid' in locals() and f_mid is not None and abs(f_mid) < min_val:
            best_ret = mid
        return best_ret
    else:
        return best_x
