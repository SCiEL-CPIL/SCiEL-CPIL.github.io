"""Nelder-Mead simplex on f(x)^2 for oscillatory root finding."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    HARD_TOL = 1e-8
    MAX_CALLS = 480

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    def obj(x):
        """Objective: f(x)^2, minimized to find root."""
        fx = safe_f(x)
        if fx != float('nan'):
            return fx * fx
        return float('inf')

    # Evaluate starting point
    f0 = safe_f(x0)
    best_x = x0
    best_obj = f0 * f0 if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_obj
        if fx != float('nan'):
            o = fx * fx
            if o < best_obj:
                best_obj = o
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # Cache for evaluated points to avoid redundant calls
    eval_cache = {x0: f0}

    def get_f(x):
        # Use a tolerance for cache lookup to handle floating point
        for k, v in eval_cache.items():
            if abs(k - x) < 1e-15 * max(1.0, abs(x)):
                return v
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    def get_obj(x):
        fx = get_f(x)
        if fx != float('nan'):
            return fx * fx
        return float('inf')

    # ------------------------------------------------------------------
    # Nelder-Mead simplex in 1D
    # ------------------------------------------------------------------
    def nelder_mead(x_start, initial_width, max_iters=150):
        """Run Nelder-Mead on obj(x) = f(x)^2 starting near x_start."""
        nonlocal best_x, best_obj
        
        # Initialize simplex: 2 vertices in 1D
        # Try a few different initial widths to handle unknown oscillation period
        widths = [initial_width]
        
        for width in widths:
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
            
            # Simplex vertices
            v1 = x_start
            v2 = x_start + width
            
            f1 = get_obj(v1)
            f2 = get_obj(v2)
            
            # Check if we already found a root
            if f1 <= HARD_TOL * HARD_TOL:
                return v1
            if f2 <= HARD_TOL * HARD_TOL:
                return v2
            
            # Ensure f1 <= f2 (v1 is better)
            if f1 > f2:
                v1, v2 = v2, v1
                f1, f2 = f2, f1
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for _ in range(max_iters):
                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                    break
                
                # Check convergence
                if f1 <= HARD_TOL * HARD_TOL:
                    return v1
                if f2 <= HARD_TOL * HARD_TOL:
                    return v2
                
                # If simplex is too small, break
                if abs(v2 - v1) < 1e-14 * max(1.0, abs(v1)):
                    break
                
                # Reflection: reflect worst vertex (v2) through best (v1)
                # In 1D: v_r = v1 - alpha * (v2 - v1) = (1+alpha)*v1 - alpha*v2
                vr = (1 + alpha) * v1 - alpha * v2
                fr = get_obj(vr)
                
                if fr < f1:
                    # Try expansion
                    ve = (1 + alpha + gamma) * v1 - (alpha + gamma) * v2
                    fe = get_obj(ve)
                    if fe < fr:
                        v2 = ve
                        f2 = fe
                    else:
                        v2 = vr
                        f2 = fr
                elif fr < f2:
                    # Accept reflection
                    v2 = vr
                    f2 = fr
                else:
                    # Contraction
                    if fr < f2:
                        # Outside contraction
                        vc = (1 + alpha + rho) * v1 - (alpha + rho) * v2
                    else:
                        # Inside contraction
                        vc = (1 + rho) * v1 + rho * (v2 - v1)
                    fc = get_obj(vc)
                    
                    if fc < min(f1, f2):
                        v2 = vc
                        f2 = fc
                    else:
                        # Shrink
                        v2 = (v1 + v2) / 2.0
                        f2 = get_obj(v2)
                
                # Reorder so v1 has smaller objective
                if f1 > f2:
                    v1, v2 = v2, v1
                    f1, f2 = f2, f1
                
                # Update global best
                if f1 < best_obj:
                    best_obj = f1
                    best_x = v1
                if f2 < best_obj:
                    best_obj = f2
                    best_x = v2
            
            # After Nelder-Mead, if we have a good result, return it
            if f1 <= HARD_TOL * HARD_TOL:
                return v1
            if f2 <= HARD_TOL * HARD_TOL:
                return v2

        return None

    # ------------------------------------------------------------------
    # Try Nelder-Mead with multiple initial widths to handle oscillation
    # ------------------------------------------------------------------
    # The oscillation period is unknown. Try several scales.
    # Start with small steps (in case root is very close) and go larger.
    
    base = max(1e-8, abs(x0) * 1e-6)
    # Try widths: base, base*10, base*100, base*1000, base*10000
    # But limit total calls
    width_factors = [1, 10, 100, 1000, 10000]
    
    found = None
    for factor in width_factors:
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break
        
        width = base * factor
        result = nelder_mead(x0, width, max_iters=100)
        
        if result is not None:
            found = result
            break
        
        # Also try from the other direction
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break
        result = nelder_mead(x0 - width, width, max_iters=100)
        if result is not None:
            found = result
            break
    
    if found is not None:
        return found

    # If Nelder-Mead didn't find a root within tolerance,
    # do a final dense scan around the best point found
    if best_x == best_x and math.isfinite(best_obj):
        # Fine scan around best_x
        local_width = max(1e-10, abs(best_x) * 1e-6)
        for i in range(-20, 21):
            if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                break
            x_try = best_x + i * local_width
            fx = get_f(x_try)
            if fx != float('nan') and abs(fx) <= HARD_TOL:
                return x_try
    
    # Return best candidate
    if best_x == best_x and math.isfinite(best_obj):
        return best_x
    return x0
# EVOLVE-BLOCK-END
