import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    max_evals = 490
    eval_count = 0
    best_x = float(x0)
    best_f = float('inf')
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_f
        if eval_count >= max_evals:
            return best_f
        if time.time() > deadline:
            return best_f
        try:
            val = f(x)
            eval_count += 1
            if not math.isfinite(val):
                # If f(x) is inf or nan, treat as large value
                return float('inf')
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x
            return val
        except Exception:
            eval_count += 1
            return float('inf')

    x0 = float(x0)
    f0 = eval_f(x0)
    if abs(f0) <= 1e-8:
        return x0

    # Initial second point for Secant
    # Choose a small step relative to x0
    h = 1e-5 * max(1.0, abs(x0))
    x1 = x0 + h
    f1 = eval_f(x1)
    if abs(f1) <= 1e-8:
        return x1

    # If f0 and f1 have opposite signs, we have a bracket immediately
    bracket = None
    if f0 * f1 < 0:
        bracket = (x0, x1)

    # Phase 1: Secant Method
    # We run Secant for up to 30 iterations or until we find a root/bracket
    x_prev = x0
    f_prev = f0
    x_curr = x1
    f_curr = f1
    
    secant_steps = 0
    max_secant_steps = 30
    
    while secant_steps < max_secant_steps:
        if time.time() > deadline or eval_count >= max_evals:
            break
            
        # Check for convergence
        if abs(f_curr) <= 1e-8:
            return x_curr
        
        # Check for bracket from previous step
        if f_prev * f_curr < 0:
            bracket = (x_prev, x_curr)
            # Switch to bisection phase if we have a bracket and secant is not converging fast?
            # Or just continue secant? Secant with a bracket is essentially Regula Falsi if we keep the side with the smaller f value.
            # Let's switch to a robust bracketing method if we've taken some steps.
            if secant_steps > 5:
                break # Switch to Phase 2

        # Compute Secant step
        if f_curr == f_prev:
            # Parallel tangent, stall. Perturb.
            x_curr = x_curr * (1 + 1e-6) + 1e-6
            f_curr = eval_f(x_curr)
            secant_steps += 1
            continue
            
        slope = (f_curr - f_prev) / (x_curr - x_prev)
        if slope == 0:
            x_curr = x_curr * (1 + 1e-6) + 1e-6
            f_curr = eval_f(x_curr)
            secant_steps += 1
            continue
            
        x_next = x_curr - f_curr / slope
        
        # Damping: If step is too large, dampen it
        # Heuristic: if |x_next - x_curr| > 10 * |x_curr - x_prev|, dampen
        step_size = abs(x_next - x_curr)
        prev_step_size = abs(x_curr - x_prev)
        if prev_step_size > 0 and step_size > 10 * prev_step_size:
            # Dampen towards previous point
            x_next = x_curr - 0.5 * (x_next - x_curr)
            
        # Also ensure x_next is not identical to previous points to avoid division by zero next time
        if abs(x_next - x_curr) < 1e-15:
            x_next = x_curr * (1 + 1e-8) + 1e-8

        f_next = eval_f(x_next)
        if abs(f_next) <= 1e-8:
            return x_next
            
        # Update
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next
        
        # Check for new bracket
        if f_prev * f_curr < 0:
            bracket = (x_prev, x_curr)
            
        secant_steps += 1

    # Phase 2: Bracketing and Bisection
    # If we have a bracket, use it. If not, find one.
    if bracket is None:
        # Try to find a bracket by expanding from x0
        a = x0
        fa = f0
        # Expand in both directions
        step = 1e-2 * max(1.0, abs(x0))
        found = False
        
        # Expand to the right
        for i in range(20):
            if time.time() > deadline or eval_count >= max_evals:
                break
            b = x0 + step * (2 ** i)
            fb = eval_f(b)
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                bracket = (a, b)
                found = True
                break
        if not found:
            # Expand to the left
            for i in range(20):
                if time.time() > deadline or eval_count >= max_evals:
                    break
                a = x0 - step * (2 ** i)
                fa = eval_f(a)
                if abs(fa) <= 1e-8:
                    return a
                # Check against original x0? No, we need a pair.
                # We have x0 and fa. Check if sign changed between x0 and a?
                # Actually, we should check against the last point on the other side or just x0.
                # Simpler: check if fa * f0 < 0
                if fa * f0 < 0:
                    bracket = (a, x0)
                    found = True
                    break
                # Also check if fa * fb (from previous right expansion) < 0? 
                # That's complicated. Let's just stick to expanding from x0.
                # If we didn't find a sign change with x0, we might have missed a root between expansions.
                # But for ill-conditioned, usually root is near x0.
                pass

    if bracket is None:
        # No bracket found. Return best guess.
        return best_x

    a, b = bracket
    fa = eval_f(a) if a != x0 and a != x1 else (f0 if a == x0 else f1)
    fb = eval_f(b) if b != x0 and b != x1 else (f0 if b == x0 else f1)
    
    # Ensure we have valid signs
    if fa * fb > 0:
        # Re-evaluate to be safe
        fa = eval_f(a)
        fb = eval_f(b)
        if fa * fb > 0:
            return best_x

    # Bisection / Illinois Method
    # Illinois is slightly faster than bisection for linear functions.
    while eval_count < max_evals and time.time() < deadline:
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
            
        # Illinois step
        if abs(fa) < abs(fb):
            # Swap to keep fa smaller? No, Illinois keeps the side with smaller f as the "new" point 
            # and moves the other one.
            # Standard Illinois:
            # If |fa| < |fb|, swap a<->b, fa<->fb
            # c = (a+b)/2
            # fc = f(c)
            # If fa*fc < 0, b=c, fb=fc
            # Else:
            #   a = 2*c - b (extrapolate)
            #   fa = f(a)
            #   If |fa| < |fb|, swap...
            
            # Simpler Bisection is safer and 1 eval/step.
            # Let's use Bisection to ensure robustness.
            pass
        
        # Bisection
        c = 0.5 * (a + b)
        fc = eval_f(c)
        if abs(fc) <= 1e-8:
            return c
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    # Return the point with the smallest |f(x)| found
    return best_x
