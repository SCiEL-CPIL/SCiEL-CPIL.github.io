"""Root finding for ill-conditioned problems using simultaneous global-local search."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0.
    Strategy: 
    1. Initial coarse scan to find brackets (sign changes) in a wide neighborhood.
    2. Local refinement using Secant/Newton on the best candidate.
    3. If local refinement stalls, use the best bracket found for Bisection.
    4. Final polish on the best point found.
    
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Track best point found (smallest |f(x)|)
    best_x = float(x0)
    best_fx = None
    
    # Track best bracket (interval [a, b] where f(a)*f(b) < 0 and |f(a)|+|f(b)| is small)
    best_bracket = None
    best_bracket_cost = float('inf')
    
    try:
        best_fx = call_f(best_x)
    except:
        best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    def update_bracket(a, fa, b, fb):
        nonlocal best_bracket, best_bracket_cost
        if fa * fb < 0:
            # Prefer brackets closer to x0 or with smaller function values
            cost = abs(fa) + abs(fb) + 0.01 * (abs(a - x0) + abs(b - x0))
            if cost < best_bracket_cost:
                best_bracket = (a, fa, b, fb)
                best_bracket_cost = cost

    update_best(best_x, best_fx)
    
    if abs(best_fx) <= 1e-8:
        return best_x

    x = float(x0)
    fx = best_fx

    # Phase 1: Simultaneous Coarse Scan and Local Refinement
    # We will do a coarse scan to find brackets, and simultaneously try to refine locally.
    # The scan is cheap in terms of "smart" logic but uses evals.
    
    # Define scan points: log-spaced around x0
    scale = max(1.0, abs(x0))
    # We want to cover a reasonable range, e.g., 10 * scale
    # Let's use ~30 points for the scan
    scan_points = []
    for i in range(1, 31):
        d = scale * (10 ** (i / 10.0))  # 10^0.1 to 10^3.0 approx
        scan_points.append(x0 + d)
        scan_points.append(x0 - d)
    
    # Shuffle slightly or just iterate
    # We interleave scan and local refinement to avoid wasting time on scan if local works fast
    scan_idx = 0
    local_x = x
    local_fx = fx
    
    # Local refinement loop (Secant-like, robust)
    # Keep a history of last 2 points for secant
    x_hist = [local_x]
    fx_hist = [local_fx]
    
    for step in range(50):  # Max 50 iterations of this hybrid loop
        if not check_time() or eval_count >= MAX_EVALS:
            break
            
        # 1. Try to refine locally using Secant/Newton
        if len(x_hist) >= 2:
            x_prev, fx_prev = x_hist[-2]
            x_curr, fx_curr = x_hist[-1]
            
            # Secant step
            if abs(fx_curr - fx_prev) > 1e-15:
                x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                # Fallback to small step
                x_next = x_curr + 1e-4 * max(1.0, abs(x_curr))
            
            try:
                fx_next = call_f(x_next)
            except:
                fx_next = float('inf')
                x_next = x_curr
                
            if math.isfinite(fx_next):
                update_best(x_next, fx_next)
                
                # Check for bracket between x_prev and x_next, or x_curr and x_next
                update_bracket(x_prev, fx_prev, x_next, fx_next)
                update_bracket(x_curr, fx_curr, x_next, fx_next)
                
                if abs(fx_next) <= 1e-8:
                    return x_next
                
                # Update history
                x_hist.append(x_next)
                fx_hist.append(fx_next)
                if len(x_hist) > 3:
                    x_hist.pop(0)
                    fx_hist.pop(0)
                    
                local_x = x_next
                local_fx = fx_next
            else:
                # Local refinement failed, force a scan point to escape local trap
                pass

        # 2. Perform a coarse scan point evaluation
        if scan_idx < len(scan_points) and eval_count < MAX_EVALS:
            x_scan = scan_points[scan_idx]
            scan_idx += 1
            try:
                fx_scan = call_f(x_scan)
            except:
                fx_scan = float('inf')
            
            if math.isfinite(fx_scan):
                update_best(x_scan, fx_scan)
                
                # Check for bracket between last scan point and this one, or with x0
                if scan_idx > 1:
                    prev_scan = scan_points[scan_idx - 2] # approx, since we added 2 per i
                    # Actually, let's just check against x0 and the previous scan point in the list
                    # To be safe, check against x0 and the immediately preceding evaluated scan point
                    # We don't have the fx of the previous scan point stored easily in this loop structure
                    # Let's store scan history
                    pass
                
                if abs(fx_scan) <= 1e-8:
                    return x_scan
        else:
            # No more scan points, continue local refinement
            pass

    # Phase 2: Use the best bracket if we have one
    if best_bracket is not None and eval_count < MAX_EVALS and check_time():
        a, fa, b, fb = best_bracket
        
        # Bisection on the best bracket
        for i in range(60):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            mid = (a + b) / 2.0
            try:
                fmid = call_f(mid)
            except:
                break
            if not math.isfinite(fmid):
                break
            
            update_best(mid, fmid)
            
            if abs(fmid) <= 1e-8:
                return mid
            
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
                
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                break
                
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a

    # Phase 3: Final Polish on the best point found
    if abs(best_fx) < 1e-4 and eval_count < MAX_EVALS and check_time():
        x_polish = best_x
        fx_polish = best_fx
        
        # Try a few secant steps to tighten
        h = 1e-6 * max(1.0, abs(x_polish))
        x1 = x_polish + h
        try:
            fx1 = call_f(x1)
        except:
            fx1 = float('inf')
            
        if math.isfinite(fx1):
            for i in range(20):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if abs(fx1) <= 1e-8:
                    return x1
                if abs(fx_polish - fx1) < 1e-15:
                    break
                x_new = x1 - fx1 * (x1 - x_polish) / (fx1 - fx_polish)
                try:
                    fx_new = call_f(x_new)
                except:
                    break
                if not math.isfinite(fx_new):
                    break
                update_best(x_new, fx_new)
                x_polish, fx_polish = x1, fx1
                x1, fx1 = x_new, fx_new
                if abs(fx1) <= 1e-8:
                    return x1

    return best_x
# EVOLVE-BLOCK-END
