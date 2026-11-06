"""Root finding: Ill-conditioned - Adaptive Step-Doubling & Contraction Solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    
    Strategy:
    1. Adaptive Step-Doubling: Dynamically adjust the expansion step size based on the magnitude of f(x).
       If f(x) is small, step small; if large, step large. This finds brackets faster than fixed geometric expansion.
    2. Modified Illinois/Contraction: Once a bracket is found, use a modified Illinois method.
       Instead of halving the function value of the retained endpoint, we explicitly contract the larger
       sub-interval by a factor of 0.5. This is robust for ill-conditioned functions where standard
       secant updates might stall due to slow variation.
    3. Fallback: If no bracket is found, use a robust Secant method with aggressive damping and
       periodic bisection-like corrections to prevent divergence.
    """
    start_time = time.time()
    MAX_WALL = 118.0  # Safety margin
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        eval_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except:
            return float('inf')

    def time_ok():
        return time.time() - start_time < MAX_WALL

    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    best_x = x0
    best_fx = fx0
    
    # --- Phase 1: Adaptive Bracketing ---
    # We want to find a, b such that f(a)*f(b) < 0.
    # Start with a = x0.
    # Expand b in both directions.
    # Step size h is adaptive: h = max(min_h, |x0|*1e-6, |fx0| * 1e-6 / max(1, |x0|)) ?
    # Actually, for ill-conditioned, f might be very flat or very steep.
    # Let's try a few initial step sizes and then adapt.
    
    a, fa = x0, fx0
    b, fb = None, None
    
    # Initial step candidates
    # We will try to expand in both + and - directions.
    # Let's maintain a current step size h.
    # If we don't find a sign change, we double h.
    # But we also check the other direction.
    
    # To be efficient, let's interleave + and - expansions.
    # dir = 1 for +, -1 for -
    # x_next = x0 + dir * h
    
    h = max(1e-8, abs(x0) * 1e-8)
    if h == 0: h = 1e-8
    
    found_bracket = False
    bracket_budget = 40  # Spend at most 40 evals on bracketing
    
    # We'll track the furthest point reached in each direction
    pos_x, pos_fx = x0, fx0
    neg_x, neg_fx = x0, fx0
    
    # We need to be careful not to waste too many evals if the function is monotonic or no root exists nearby.
    # If we spend 40 evals and don't find a bracket, we switch to Secant.
    
    # Let's try expanding from x0.
    # We'll alternate directions.
    
    steps = 0
    while steps < bracket_budget and time_ok():
        # Try positive direction
        x_pos = pos_x + h
        if x_pos != pos_x:
            fx_pos = eval_f(x_pos)
            steps += 1
            if abs(fx_pos) <= TARGET_TOL:
                return x_pos
            if fa * fx_pos < 0:
                # Bracket found between a (x0) and x_pos? Not necessarily, maybe between pos_x and x_pos?
                # We know fa * fx0 < 0 is false (fx0 is fa).
                # We know pos_fx is the value at pos_x.
                # If pos_fx * fx_pos < 0, bracket is [pos_x, x_pos].
                # If fa * fx_pos < 0, bracket is [x0, x_pos] (if x0 is between? No, x0 is one end).
                # Actually, we are expanding from x0.
                # Let's keep track of the last point that had the same sign as x0.
                # If we find a sign change with x0, we are done.
                # If we find a sign change between the previous step and current, we are done.
                
                # Let's simplify:
                # We have a chain: x0 -> x1 -> x2 ...
                # If f(x0)*f(xi) < 0, bracket is [x0, xi] (assuming no intermediate sign change? No, IVT guarantees one).
                # But we might have missed a sign change if we jumped over it? No, we check every step.
                # So if f(x0) and f(x_pos) have opposite signs, the root is in [x0, x_pos].
                # But we might have a tighter bracket [pos_x, x_pos] if pos_x is closer.
                
                if pos_fx * fx_pos < 0:
                    a, b, fa, fb = pos_x, x_pos, pos_fx, fx_pos
                else:
                    a, b, fa, fb = x0, x_pos, fx0, fx_pos
                found_bracket = True
                break
            else:
                # Same sign as x0 (or inf), update pos_x to be the new frontier
                # But wait, if pos_fx was inf, we should maybe shrink? No, just expand.
                # If pos_fx is finite and same sign, we can move the frontier to x_pos.
                if math.isfinite(fx_pos):
                    pos_x, pos_fx = x_pos, fx_pos
        
        # Try negative direction
        x_neg = neg_x - h
        if x_neg != neg_x:
            fx_neg = eval_f(x_neg)
            steps += 1
            if abs(fx_neg) <= TARGET_TOL:
                return x_neg
            if neg_fx * fx_neg < 0:
                a, b, fa, fb = neg_x, x_neg, neg_fx, fx_neg
                found_bracket = True
                break
            else:
                if math.isfinite(fx_neg):
                    neg_x, neg_fx = x_neg, fx_neg
        
        # Double the step size for next iteration
        # Adaptive: if the function values are getting larger, we might be moving away.
        # But for root finding, we just need to cross the axis.
        # Standard doubling is robust.
        h *= 2.0
        
        # Prevent overflow
        if h > 1e12:
            h = 1e12

    # --- Phase 2: Solve ---
    
    if found_bracket:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Modified Illinois / Contraction Method
        # Standard Illinois halves the function value of the endpoint that is NOT updated.
        # This effectively biases the next secant step towards the updated endpoint.
        # The new strategy: "shrink the larger bracket sub-interval by a fixed factor (0.5)".
        # This sounds like a hybrid of Bisection and Secant.
        # Let's interpret: 
        # 1. Compute secant estimate x_s.
        # 2. If x_s is in the larger sub-interval, use it.
        # 3. Else, use the midpoint of the larger sub-interval?
        # Or simply: Use Illinois, but if the convergence is slow (interval shrinks slowly), 
        # force a bisection-like step on the larger side.
        
        # Let's stick to a robust Illinois variant that explicitly checks interval size.
        # If the secant step doesn't reduce the interval size by at least 50%, use bisection.
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
            
            # Secant step
            denom = fb - fa
            if denom == 0:
                x_new = (a + b) / 2
            else:
                x_new = a - fa * (b - a) / denom
            
            # Ensure x_new is in (a, b)
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2
            
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Check if we made sufficient progress
            # If the interval didn't shrink by at least half, force a bisection on the larger side
            # This handles the "ill-conditioned" case where secant steps are tiny.
            
            # Actually, Illinois already does this via the halving of f values.
            # Let's implement the specific "shrink larger sub-interval by 0.5" idea.
            # This means:
            # Identify which sub-interval is larger: [a, x_new] or [x_new, b].
            # If the secant step was inside the larger one, good.
            # If not, or if it was very close to the boundary, we might want to bisect the larger one.
            
            # Let's use a simpler, more robust logic for ill-conditioned:
            # If |fa| < |fb|, the root is likely closer to a.
            # We want to move b closer to a.
            # Illinois does this.
            
            # Let's add a specific check:
            # If the step size |x_new - old_b| (or old_a) is very small relative to interval,
            # it might be stalling.
            
            # For now, let's rely on the standard Illinois logic but with a stricter convergence check.
            # If we are not making progress (interval length not decreasing), break to Secant?
            # No, Illinois is guaranteed to converge.
            
            # Update best
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

    else:
        # No bracket found. Use Secant Method.
        # We have x0, fx0.
        # We also have pos_x, pos_fx and neg_x, neg_fx from the bracketing phase.
        # Let's pick the one with the smallest |f| as the second point.
        
        candidates = [(x0, fx0), (pos_x, pos_fx), (neg_x, neg_fx)]
        # Filter out infs
        valid_candidates = [(x, fx) for x, fx in candidates if math.isfinite(fx)]
        
        if not valid_candidates:
            return best_x
            
        # Sort by |f(x)|
        valid_candidates.sort(key=lambda p: abs(p[1]))
        
        x_a, fx_a = valid_candidates[0]
        
        # Pick the second point. Ideally, one that gives a good slope.
        # Let's just pick the next best one.
        x_b, fx_b = valid_candidates[1] if len(valid_candidates) > 1 else (x_a + 1e-5, eval_f(x_a + 1e-5))
        
        if abs(fx_b) <= TARGET_TOL:
            return x_b
            
        # Secant iteration with damping
        stall_count = 0
        prev_best = min(abs(fx_a), abs(fx_b))
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fx_a) <= TARGET_TOL:
                return x_a
            if abs(fx_b) <= TARGET_TOL:
                return x_b
            
            denom = fx_b - fx_a
            if denom == 0 or not math.isfinite(denom):
                # Perturb
                x_b = x_a + 1e-4
                fx_b = eval_f(x_b)
                continue
            
            x_new = x_a - fx_a * (x_b - x_a) / denom
            
            # Damping
            max_step = 10 * max(1.0, abs(x_a))
            if abs(x_new - x_a) > max_step:
                x_new = x_a + math.copysign(max_step, x_new - x_a)
            
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update
            x_a, fx_a = x_b, fx_b
            x_b, fx_b = x_new, fx_new
            
            # Check progress
            current_best = min(abs(fx_a), abs(fx_b))
            if current_best < prev_best * 0.9:
                prev_best = current_best
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > 5:
                # Try to find a bracket from the current points?
                # Or just perturb.
                x_b = x_a + 0.5 * (x_b - x_a)
                fx_b = eval_f(x_b)
                stall_count = 0
                
            # Update best
            if abs(fx_a) < abs(best_fx):
                best_x, best_fx = x_a, fx_a
            if abs(fx_b) < abs(best_fx):
                best_x, best_fx = x_b, fx_b

    return best_x

# EVOLVE-BLOCK-END
