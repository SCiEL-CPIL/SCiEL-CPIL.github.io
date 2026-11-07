"""Root finder for 3 fixed oscillatory equations using global multi-scale grid + Illinois method."""
import math
import time


# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin for 120s limit
    MAX_CALLS = 495     # Stay under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _time_exceeded():
        return time.time() - start_time > TIME_LIMIT

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if _time_exceeded():
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not isinstance(val, (int, float)):
            return float('nan')
        val = float(val)
        if not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return val

    def _illinois(a, b, fa, fb, max_iter=30):
        """Illinois method to find root in [a, b] where f(a)*f(b) < 0."""
        nonlocal calls, best_x, best_abs_f
        if _time_exceeded() or calls >= MAX_CALLS - 5:
            return None
        
        # Ensure fa and fb are valid
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return None
        if fa * fb >= 0:
            return None
        
        # Illinois method
        for _ in range(max_iter):
            if _time_exceeded() or calls >= MAX_CALLS - 5:
                break
            # Compute interpolation point
            if abs(fa) < abs(fb):
                # c is closer to a
                c = a + (b - a) * fa / (fa - fb)
            else:
                # c is closer to b
                c = b + (a - b) * fb / (fa - fb)
            
            fc = _f(c)
            if math.isnan(fc):
                break
            
            if abs(fc) <= 1e-8:
                return c
            
            if fa * fc < 0:
                # Root in [a, c]
                b = c
                fb = fc
                fa = fa * 0.5  # Illinois damping
            else:
                # Root in [c, b]
                a = c
                fa = fc
                fb = fb * 0.5  # Illinois damping
            
            if abs(b - a) < 1e-12:
                break
        
        # Return best point in the final bracket
        if math.isfinite(fa) and math.isfinite(fb):
            if abs(fa) < abs(fb):
                return a
            else:
                return b
        return None

    # Phase 1: Evaluate at x0
    x = float(x0)
    fx = _f(x)
    if abs(fx) <= 1e-8:
        return x

    # Phase 2: Global multi-scale grid search to find a bracket
    # Strategy: Exponentially spaced points across a wide domain
    # to identify sign-change intervals.
    
    bracket_found = False
    bracket = None
    
    # Define search radii (exponentially increasing)
    radii = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    
    for radius in radii:
        if _time_exceeded() or calls >= MAX_CALLS - 30:
            break
        
        # Number of points: more points for smaller radii (finer resolution near x0)
        # Scale number of points with log of radius
        n_points = max(10, min(60, int(10 * math.log2(radius + 1) + 10)))
        
        lower = x - radius
        upper = x + radius
        
        prev_x = None
        prev_f = None
        
        for i in range(n_points + 1):
            if _time_exceeded() or calls >= MAX_CALLS - 30:
                break
            
            cur_x = lower + i * (upper - lower) / n_points
            cur_f = _f(cur_x)
            
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue
            
            if abs(cur_f) <= 1e-8:
                return cur_x
            
            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    bracket = (prev_x, cur_x)
                    bracket_found = True
                    break
            
            prev_x = cur_x
            prev_f = cur_f
        
        if bracket_found:
            break
    
    # Phase 3: Apply Illinois method if a bracket was found
    if bracket_found:
        a, b = bracket
        fa = _f(a)
        fb = _f(b)
        
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            root = _illinois(a, b, fa, fb)
            if root is not None and math.isfinite(root):
                f_root = _f(root)
                if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                    return root
                if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                    best_x = root
                    best_abs_f = abs(f_root)
        else:
            # Re-evaluate if needed
            if math.isnan(fa):
                fa = _f(a)
            if math.isnan(fb):
                fb = _f(b)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
                root = _illinois(a, b, fa, fb)
                if root is not None and math.isfinite(root):
                    f_root = _f(root)
                    if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                        return root
                    if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                        best_x = root
                        best_abs_f = abs(f_root)
    
    # Phase 4: If no bracket found, try a finer local scan around x0
    if best_abs_f > 1e-8 and not _time_exceeded() and calls < MAX_CALLS - 20:
        # Try a very fine local scan
        local_radius = 0.5
        n_local = 50
        lower = x - local_radius
        upper = x + local_radius
        
        prev_x = None
        prev_f = None
        
        for i in range(n_local + 1):
            if _time_exceeded() or calls >= MAX_CALLS - 10:
                break
            
            cur_x = lower + i * (upper - lower) / n_local
            cur_f = _f(cur_x)
            
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue
            
            if abs(cur_f) <= 1e-8:
                return cur_x
            
            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    a, b = prev_x, cur_x
                    fa, fb = prev_f, cur_f
                    root = _illinois(a, b, fa, fb)
                    if root is not None and math.isfinite(root):
                        f_root = _f(root)
                        if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                            return root
                        if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                            best_x = root
                            best_abs_f = abs(f_root)
                    break
            
            prev_x = cur_x
            prev_f = cur_f
    
    # Phase 5: Try even wider search if still no good result
    if best_abs_f > 1e-8 and not _time_exceeded() and calls < MAX_CALLS - 15:
        # Try a coarse wide scan
        wide_radius = 1000.0
        n_wide = 30
        lower = x - wide_radius
        upper = x + wide_radius
        
        prev_x = None
        prev_f = None
        
        for i in range(n_wide + 1):
            if _time_exceeded() or calls >= MAX_CALLS - 5:
                break
            
            cur_x = lower + i * (upper - lower) / n_wide
            cur_f = _f(cur_x)
            
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue
            
            if abs(cur_f) <= 1e-8:
                return cur_x
            
            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    a, b = prev_x, cur_x
                    fa, fb = prev_f, cur_f
                    root = _illinois(a, b, fa, fb)
                    if root is not None and math.isfinite(root):
                        f_root = _f(root)
                        if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                            return root
                        if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                            best_x = root
                            best_abs_f = abs(f_root)
                    break
            
            prev_x = cur_x
            prev_f = cur_f
    
    return best_x

# EVOLVE-BLOCK-END
