"""
Strategy: Global Best-First Root Finder
1. Coarse adaptive grid sampling to identify brackets [a, b] where f(a)*f(b) < 0.
2. For each bracket, use a robust 1D minimization/root-finding hybrid (Illinois method variant)
   which is derivative-free and robust to oscillations.
3. Prioritize brackets with the smallest width or highest potential to converge quickly.
4. Strict call budgeting and time guards.
"""
import math
import time
import sys

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Leave 10s safety margin for 120s limit
    
    # State tracking
    calls = 0
    best_x = None
    best_f_val = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_f_val
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Finite check
        try:
            val = float(f(x))
        except Exception:
            return float('nan')
        
        if not math.isfinite(val):
            return float('nan')
            
        calls += 1
        
        if abs(val) < abs(best_f_val):
            best_f_val = val
            best_x = x
            
        return val

    # Initial evaluation
    x = float(x0)
    fx = _f(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Data structure for brackets: list of (a, b, fa, fb)
    brackets = []
    
    # Phase 1: Coarse Adaptive Search for Brackets
    # We expand from x0 in both directions with increasing step sizes.
    # For oscillatory functions, we need to sample frequently enough to catch sign changes.
    # Start with a small step, then grow.
    
    # Initial points
    a = x
    fa = fx
    
    # Search right
    step = 1e-3
    b = x + step
    fb = _f(b)
    
    # If no sign change, expand
    # We want to find a bracket. If we find one, we add it to the list.
    # We continue searching for more brackets if we have time and calls to spare,
    # but usually one good bracket is enough if we use a robust method.
    # However, for oscillatory functions, x0 might be far from a root, or in a "quiet" zone.
    
    # Strategy:
    # 1. Try to find a bracket in [x0, x0+R] and [x0-R, x0]
    # 2. If found, solve it.
    # 3. If not found, expand R.
    
    # Let's implement a robust bracket finder that returns the first valid bracket found.
    # To handle oscillations, we use a moderate step size initially.
    
    def find_bracket(x_start, f_start, direction, max_steps=100):
        """
        Expands from x_start in 'direction' (1 or -1) to find a bracket.
        Returns (a, b, fa, fb) or None.
        """
        a = x_start
        fa = f_start
        b = x_start + direction * 1e-3
        fb = _f(b)
        
        if math.isnan(fb):
            return None
            
        if fa * fb < 0:
            return (a, b, fa, fb)
            
        # Geometric expansion
        step = 1e-3
        for _ in range(max_steps):
            if time.time() - start_time > TIME_LIMIT:
                return None
            if calls > 400: # Safety call limit for bracketing phase
                return None
                
            step *= 1.5
            b = x_start + direction * step
            fb = _f(b)
            
            if math.isnan(fb):
                # Try to shrink or skip? For robustness, just stop this direction.
                return None
                
            if fa * fb < 0:
                return (a, b, fa, fb)
                
            # Update a to b to maintain continuity for next check?
            # Actually, for oscillatory, if we jump over a root, we might miss it.
            # Better: keep a fixed "left" and expand "right"?
            # No, if f is oscillatory, f(x_start) and f(x_start + large) might have same sign
            # even if roots exist in between.
            # To be safe against oscillations, we should sample intermediate points.
            
            # Revised strategy for find_bracket:
            # Instead of just checking endpoints, check intermediate points if step grows large.
            # But for simplicity and speed, let's assume the "first" root found by geometric expansion
            # is the target. If the function is highly oscillatory, this might fail.
            # The task says "3 fixed oscillatory equations".
            # If oscillations are fast, 1.5x expansion might skip roots.
            # Let's add a check: if step > 1.0, sample the midpoint of previous and current b.
            
            if step > 1.0:
                mid = x_start + direction * (step / 1.5) # Previous b
                fmid = _f(mid)
                if not math.isnan(fmid):
                    if fa * fmid < 0:
                        return (a, mid, fa, fmid)
                    if fmid * fb < 0:
                        return (mid, b, fmid, fb)
                        
            a = b
            fa = fb
            
        return None

    # Try right first
    bracket = find_bracket(x, fx, 1)
    if bracket is None:
        # Try left
        bracket = find_bracket(x, fx, -1)
        
    if bracket is None:
        # Fallback: If no bracket found, return best_x found so far (likely failure, but best effort)
        # Or try a wider, coarser search
        # For now, return best_x
        if best_x is not None:
            return best_x
        return x

    a, b, fa, fb = bracket
    
    # Phase 2: Solve within [a, b] using Illinois Method (robust, derivative-free)
    # Illinois is a variant of bisection that moves the endpoint with the larger function value
    # closer to the other, effectively doing a weighted bisection.
    # It is very robust for oscillatory functions as long as the bracket is tight enough.
    
    # Ensure fa < 0 < fb or vice versa for consistent logic
    # Illinois doesn't strictly require ordering, just sign change.
    
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    # Max iterations for Illinois
    max_iter = 50
    for i in range(max_iter):
        if time.time() - start_time > TIME_LIMIT:
            break
        if calls > 450:
            break
            
        # Check convergence
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
        
        # Illinois step
        # If |f1| > |f2|, replace x1 with midpoint, else replace x2
        if abs(f1) > abs(f2):
            x1 = 0.5 * (x1 + x2)
            f1 = _f(x1)
        else:
            x2 = 0.5 * (x1 + x2)
            f2 = _f(x2)
            
        if math.isnan(f1) or math.isnan(f2):
            break
            
        # Update best
        if abs(f1) < abs(best_f_val):
            best_f_val = f1
            best_x = x1
        if abs(f2) < abs(best_f_val):
            best_f_val = f2
            best_x = x2
            
        # If bracket is lost (shouldn't happen with Illinois if no NaNs), break
        if f1 * f2 > 0:
            break

    # Final check
    if abs(f1) <= abs(f2):
        if abs(f1) <= 1e-8:
            return x1
        return x1
    else:
        if abs(f2) <= 1e-8:
            return x2
        return x2

# EVOLVE-BLOCK-END
