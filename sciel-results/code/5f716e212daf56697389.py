import math
import time
import random

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin for 150s kill limit
    
    calls = 0
    best_x = x0
    best_f_abs = float('inf')
    cache = {}
    
    def _f(x):
        nonlocal calls, best_x, best_f_abs
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        if x in cache:
            return cache[x]
            
        try:
            val = float(f(x))
        except Exception:
            cache[x] = float('nan')
            return float('nan')
            
        if not math.isfinite(val):
            cache[x] = float('nan')
            return float('nan')
            
        calls += 1
        cache[x] = val
        
        abs_val = abs(val)
        if abs_val < best_f_abs:
            best_f_abs = abs_val
            best_x = x
            
        return val

    x0 = float(x0)
    fx0 = _f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Stochastic Hill-Climbing for Bracketing + Secant Refinement
    # 1. Use a random walk to explore the function space.
    #    This avoids the "aliasing" problem of fixed grids where the step size
    #    might be commensurate with the oscillation period, missing sign changes.
    #    We start at x0 and take random steps, shrinking the step size as we
    #    find smaller |f(x)| values, mimicking a coarse-to-fine search.
    # 2. Once a bracket [a, b] with f(a)*f(b) < 0 is found, use the Secant method.
    #    Secant method is faster than Bisection (superlinear convergence) which is
    #    critical for minimizing mean_calls in the scoring formula.
    
    # Parameters for the stochastic search
    # We want to spend roughly 300-400 calls on search, leaving 100-200 for refinement.
    # Secant method typically needs < 20 calls to reach 1e-8 precision from a tight bracket.
    # So we have a generous budget for the search phase.
    
    search_budget = 400
    current_x = x0
    current_fx = fx0
    step_size = 1.0  # Initial step size, will adapt
    
    # We keep a history of the last few points to check for sign changes
    # and to help the random walk "climb" towards roots.
    # For simplicity and robustness, we'll just do a random walk with adaptive step size.
    
    a_bracket = None
    b_bracket = None
    fa_bracket = None
    fb_bracket = None
    bracket_found = False
    
    # To make the random walk more effective, we can bias it towards regions
    # where |f(x)| is small. However, a simple random walk with step size
    # reduction is often sufficient for oscillatory functions.
    
    # Let's use a "Levy Flight" style or just adaptive Gaussian steps.
    # Simple approach: 
    # - Start with large steps to explore global structure.
    # - When |f(x)| decreases, reduce step size to explore locally.
    # - When |f(x)| increases, increase step size to escape local minima.
    
    # We also need to handle the case where the root is far from x0.
    # The random walk should be able to cover a wide range.
    
    # Let's try a mix: 
    # 1. A few large random jumps to cover [-1000, 1000] relative to x0?
    #    No, that's too expensive. 
    #    The current grid approach covers [-10000, 10000]. 
    #    A random walk with 400 steps and average step size 10 would cover ~4000 distance.
    #    Let's ensure the step size is large enough initially.
    
    # Adaptive Step Size Logic:
    # step_size = max(min_step, current_step * factor)
    # If |f(new)| < |f(old)|: factor = 0.5 (move closer)
    # If |f(new)| > |f(old)|: factor = 1.5 (move away/explore)
    # But this can get stuck. 
    # Better: Random step direction and magnitude from a distribution.
    
    # Let's use a simple random walk with a decaying step size envelope.
    # Step i has magnitude U(0, S_i) where S_i decays.
    # But we also want to jump.
    
    # Alternative: Random sampling in expanding rings?
    # Let's stick to a robust random walk that biases towards smaller |f|.
    
    # Initialize
    x_curr = x0
    f_curr = fx0
    step_scale = 1.0
    
    # We will run the search until budget or time limit.
    # To check for sign change, we need the previous point.
    x_prev = x_curr
    f_prev = f_curr
    
    for i in range(search_budget):
        if calls >= search_budget or time.time() - start_time > TIME_LIMIT:
            break
            
        # Determine step size
        # Decay the maximum step size over time to focus on local regions
        # but keep it large enough to escape local traps.
        # max_step = 1000 / (1 + i/50) ?
        # Let's use a fixed schedule: 
        # First 100 steps: large steps (up to 100)
        # Next 100 steps: medium steps (up to 10)
        # Next 100 steps: small steps (up to 1)
        # Last 100 steps: tiny steps (up to 0.1)
        
        if i < 100:
            max_step = 100.0
        elif i < 200:
            max_step = 10.0
        elif i < 300:
            max_step = 1.0
        else:
            max_step = 0.1
            
        # Random step direction and magnitude
        # Uniform distribution in [-max_step, max_step]
        step = random.uniform(-max_step, max_step)
        x_new = x_curr + step
        
        f_new = _f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Check for sign change between x_prev and x_new
        # Note: x_curr is the point we are moving FROM. 
        # The random walk jumps from x_curr to x_new.
        # If f(x_curr) and f(x_new) have opposite signs, we have a bracket.
        if f_curr * f_new < 0:
            bracket_found = True
            a_bracket = x_curr
            b_bracket = x_new
            fa_bracket = f_curr
            fb_bracket = f_new
            break
            
        # Update current state
        # We might want to move to x_new only if it's an improvement, 
        # or always? 
        # If we always move, we perform a random walk.
        # If we only move on improvement, we perform a hill climb.
        # Hill climbing gets stuck in local minima of |f(x)| which are not roots.
        # Random walk is better for exploration.
        # Let's do a random walk, but bias the next step slightly towards 
        # the direction of the last improvement? 
        # For simplicity, pure random walk with the step size schedule above.
        
        x_curr = x_new
        f_curr = f_new
        
        # Update best
        if abs(f_new) < best_f_abs:
            best_f_abs = abs(f_new)
            best_x = x_new
            
        # If we hit the time limit or call limit, break
        if time.time() - start_time > TIME_LIMIT:
            break

    if bracket_found:
        # Secant Method Refinement
        # Secant method: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        # It requires two initial points. We have a_bracket and b_bracket.
        # However, Secant method is not guaranteed to converge if the function is not well-behaved,
        # but for smooth oscillatory functions near a root, it's very fast.
        # We should fall back to Bisection if Secant diverges or takes too long.
        
        x0_sec = a_bracket
        x1_sec = b_bracket
        f0_sec = fa_bracket
        f1_sec = fb_bracket
        
        # Ensure we don't divide by zero or near-zero
        if abs(f1_sec - f0_sec) < 1e-15:
            # Fall back to bisection
            x = (a_bracket + b_bracket) / 2.0
            return x
            
        for _ in range(50):
            if calls >= 495 or time.time() - start_time > TIME_LIMIT:
                break
                
            # Calculate next point
            denom = f1_sec - f0_sec
            if abs(denom) < 1e-15:
                # Degenerate case, use bisection step
                x_new = (x0_sec + x1_sec) / 2.0
                f_new = _f(x_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                # Update to keep bracketing? Secant doesn't maintain bracket.
                # If we lose the bracket, we might diverge.
                # Let's check if we still have a sign change with the new point.
                if f0_sec * f_new < 0:
                    x1_sec = x0_sec
                    f1_sec = f0_sec
                    x0_sec = x_new
                    f0_sec = f_new
                elif f1_sec * f_new < 0:
                    # x0_sec and x_new bracket the root
                    # Update x1 to x0, x0 to x_new? 
                    # Actually, let's just switch to Bisection if we lose the bracket 
                    # or if Secant is not converging.
                    # For robustness, let's check convergence.
                    if abs(x_new - x1_sec) < 1e-12:
                        return x_new
                    # If no bracket, we are in trouble. Return best effort or midpoint.
                    # Let's try to re-bracket with bisection from the last known bracket.
                    # This is complex. Let's just use Bisection from the start if Secant is risky.
                    # Actually, Secant is very fast. Let's stick to it but with a safeguard.
                    # If the new point is outside the original bracket, we might be diverging.
                    # Let's check if x_new is within [min(a,b), max(a,b)] expanded slightly.
                    # If not, fallback to Bisection.
                    
                    # Simple safeguard: If Secant step is too large, use Bisection.
                    mid = (x0_sec + x1_sec) / 2.0
                    f_mid = _f(mid)
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if f0_sec * f_mid < 0:
                        x1_sec = x0_sec
                        f1_sec = f0_sec
                        x0_sec = mid
                        f0_sec = f_mid
                    else:
                        # f1_sec * f_mid < 0 should be true if f0*f1<0
                        x0_sec = mid
                        f0_sec = f_mid
                    continue
                    
            x_new = x1_sec - f1_sec * (x1_sec - x0_sec) / denom
            f_new = _f(x_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
                
            # Check for convergence
            if abs(x_new - x1_sec) < 1e-12:
                return x_new
                
            # Update points
            x0_sec = x1_sec
            f0_sec = f1_sec
            x1_sec = x_new
            f1_sec = f_new
            
            # Safety check: If we are wandering far away, stop.
            # The original bracket width was W = abs(b-a).
            # If |x_new - (a+b)/2| > 10*W, we are likely diverging.
            # Let's just rely on the call limit and time limit.
            
        # Return the last computed point or the midpoint of the last two points
        return x1_sec

    # Fallback: Return best_x found during the sweep
    return best_x

# EVOLVE-BLOCK-END
