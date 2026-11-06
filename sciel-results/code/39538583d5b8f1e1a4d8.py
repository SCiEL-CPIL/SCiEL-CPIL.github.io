"""Root finding for ill-conditioned problems using adaptive bracketing."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using adaptive global search + hybrid refinement.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    MAX_EVALS = 500
    eval_count = 0
    
    # Cache for f evaluations to avoid redundant calls
    f_cache = {}
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        # Quantize x slightly to improve cache hit rate for near-identical calls
        key = round(x, 15)
        if key in f_cache:
            return f_cache[key]
        
        fx = f(x)
        f_cache[key] = fx
        eval_count += 1
        return fx
    
    def check_time():
        return (time.time() - start_time) < DEADLINE

    # Track best point found (lowest |f(x)|)
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except Exception:
        pass

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    # 1. Initial check
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # 2. Adaptive Bracketing Search
    # We want to find [a, b] such that f(a) and f(b) have opposite signs.
    # Start with a small step, expand exponentially if no sign change.
    # Also sample locally to catch flat roots or near roots.
    
    a = float(x0)
    fa = best_fx
    if fa is None:
        return best_x # f(x0) failed? Unlikely but safe.

    # Try to find a sign change in the vicinity
    # Strategy: 
    # 1. Check x0 + small_step, x0 - small_step
    # 2. If no sign change, expand step size
    # 3. If no sign change after large expansion, fall back to local minimization of |f(x)|
    
    step = 1e-3 * max(1.0, abs(x0))
    found_bracket = False
    bracket_a, bracket_b = None, None
    fa_bracket = None
    fb_bracket = None
    
    # Phase 1: Local exploration for sign change
    # Check immediate neighbors
    candidates = [x0 + step, x0 - step]
    
    # If x0 is very close to 0, use absolute step
    if abs(x0) < 1e-3:
         step = 1e-3
         candidates = [x0 + step, x0 - step, x0 + 10*step, x0 - 10*step]

    for c in candidates:
        if not check_time() or eval_count >= MAX_EVALS - 10:
            break
        try:
            fc = call_f(c)
            update_best(c, fc)
            if abs(fc) <= 1e-8:
                return c
            if fa * fc < 0:
                found_bracket = True
                if a < c:
                    bracket_a, bracket_b = a, c
                    fa_bracket, fb_bracket = fa, fc
                else:
                    bracket_a, bracket_b = c, a
                    fa_bracket, fb_bracket = fc, fa
                break
        except Exception:
            continue

    # Phase 2: Expand if no local sign change found
    if not found_bracket:
        step = 1e-3 * max(1.0, abs(x0))
        # Try expanding in both directions
        for direction in [1, -1]:
            if not check_time() or eval_count >= MAX_EVALS - 10:
                break
            for _ in range(20): # Logarithmic expansion
                x_try = x0 + direction * step
                try:
                    f_try = call_f(x_try)
                    update_best(x_try, f_try)
                    if abs(f_try) <= 1e-8:
                        return x_try
                    if fa * f_try < 0:
                        found_bracket = True
                        if x0 < x_try:
                            bracket_a, bracket_b = x0, x_try
                            fa_bracket, fb_bracket = fa, f_try
                        else:
                            bracket_a, bracket_b = x_try, x0
                            fa_bracket, fb_bracket = f_try, fa
                        break
                except Exception:
                    break
                step *= 2.0
            if found_bracket:
                break

    # 3. Refinement
    if found_bracket:
        # Use Illinois Algorithm (modified Regula Falsi) for robust convergence
        a, b = bracket_a, bracket_b
        fa, fb = fa_bracket, fb_bracket
        
        # Ensure fa and fb are finite
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return best_x

        for _ in range(100):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            
            # Check endpoints again (in case of floating point issues)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Illinois Method:
            # c = (a*fb - b*fa) / (fb - fa)
            # If |fa| < |fb|, shrink [a, c] by setting b = c and fb = fb/2 (or similar damping)
            # Standard Illinois:
            if abs(fa) < abs(fb):
                # Shrink the side with larger function value
                b = a
                fb = fa
                fa = fa / 2.0 # Damping factor, often 0.5
            else:
                a = b
                fa = fb
                fb = fb / 2.0
            
            # Recalculate bracket values? No, Illinois updates the *value* used for interpolation
            # But we need actual f(a) and f(b) for the next step?
            # Actually, Illinois algorithm typically keeps the actual function values at the endpoints
            # but uses a modified value for the interpolation formula to force convergence.
            
            # Let's stick to a simpler, more standard Regula Falsi with a stagnation check
            # or just Secant/Bisection hybrid.
            
            # Let's use a robust hybrid:
            # 1. Try Secant step
            # 2. If it doesn't reduce the interval or stays outside, use Bisection
            
            # Current interval [a, b] with f(a)=fa, f(b)=fb
            # Secant point
            denom = fb - fa
            if denom == 0:
                c = (a + b) / 2.0
            else:
                c = (a * fb - b * fa) / denom
            
            # Check if c is within [a, b]
            if not (a < c < b):
                c = (a + b) / 2.0
            
            try:
                fc = call_f(c)
            except Exception:
                break
            
            update_best(c, fc)
            if abs(fc) <= 1e-8:
                return c
            
            # Update bracket
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            # Stagnation check for Regula Falsi:
            # If the same endpoint is kept for too many iterations, switch to bisection
            # We can track this by checking if the interval is shrinking very slowly
            # For simplicity, let's just run it. Bisection is guaranteed to converge.
            # To ensure convergence, we can force bisection every N steps if interval doesn't shrink significantly.
            # However, 100 iterations of bisection is more than enough for 1e-8 precision from a reasonable bracket.
            
    else:
        # No bracket found. Fall back to local minimization of |f(x)|
        # Use a simple golden section search or Nelder-Mead on 1D?
        # Since we are in 1D, we can do a local search for the minimum of |f(x)|.
        # We already have best_x. Let's try to refine it with a few secant steps
        # hoping to get closer, even if we don't cross zero.
        
        # Try a few secant steps from best_x
        x = best_x
        fx = best_fx
        if fx is None:
            return best_x
            
        # Need a second point for secant
        h = 1e-6 * max(1.0, abs(x))
        try:
            x1 = x + h
            fx1 = call_f(x1)
        except Exception:
            return best_x
        
        # Secant iterations
        for _ in range(20):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx1) < abs(fx):
                x, fx = x1, fx1
                # Need new x1
                h = 1e-6 * max(1.0, abs(x))
                try:
                    x_new = x + h
                    fx_new = call_f(x_new)
                    x1, fx1 = x_new, fx_new
                except Exception:
                    break
            else:
                x, fx = x1, fx1
                h = 1e-6 * max(1.0, abs(x))
                try:
                    x_new = x - h # Try opposite direction
                    fx_new = call_f(x_new)
                    x1, fx1 = x_new, fx_new
                except Exception:
                    break
            
            if abs(fx) <= 1e-8:
                return x
                
            update_best(x, fx)

    return best_x
# EVOLVE-BLOCK-END
