import math
import time
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin for 150s limit
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    # Track evaluations
    eval_count = 1
    
    # Helper to check time
    def timed_out():
        return time.time() > deadline

    # If we already have a root, return
    if best_abs_fx <= 1e-8:
        return best_x

    # Strategy:
    # 1. Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # 2. If found, use Illinois method (robust bisection/secant hybrid) to converge.
    # 3. If no bracket is found after a few attempts, fall back to local minimization of |f(x)|
    #    or just return the best found so far.
    
    # Initial bracket search
    # Start with a small interval around x0
    a = x - 1e-3
    b = x + 1e-3
    fa = f(a)
    fb = f(b)
    eval_count += 2
    
    if timed_out():
        return best_x

    # Update best if a or b is better
    if abs(fa) < best_abs_fx:
        best_x = a
        best_fx = fa
        best_abs_fx = abs(fa)
    if abs(fb) < best_abs_fx:
        best_x = b
        best_fx = fb
        best_abs_fx = abs(fb)

    # Check for sign change
    bracket_found = False
    if not (math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb)):
        if fa * fb < 0:
            bracket_found = True
        elif fa == 0:
            return a
        elif fb == 0:
            return b

    # If no bracket, try expanding the interval
    if not bracket_found:
        step_size = 1e-3
        for _ in range(20):
            if timed_out():
                break
            
            # Try expanding left
            a_new = x - step_size
            fa_new = f(a_new)
            eval_count += 1
            if timed_out(): break
            
            if not (math.isnan(fa_new) or math.isinf(fa_new)):
                if abs(fa_new) < best_abs_fx:
                    best_x = a_new
                    best_fx = fa_new
                    best_abs_fx = abs(fa_new)
                
                # Check sign change between a_new and x (using current best_fx? No, use f(x) which is best_fx initially, but x might have moved? 
                # Actually, we should check against the original x0 or the current best? 
                # Standard bracketing: check f(a_new) * f(x) < 0
                # We have f(x) stored in best_fx only if x is best. Let's keep a separate fx0 or just use f(x) if we haven't moved x.
                # In this loop, x is still x0. So f(x) is the initial best_fx.
                if fa_new * best_fx < 0:
                    a = a_new
                    b = x
                    fa = fa_new
                    fb = best_fx
                    bracket_found = True
                    break
            
            # Try expanding right
            b_new = x + step_size
            fb_new = f(b_new)
            eval_count += 1
            if timed_out(): break
            
            if not (math.isnan(fb_new) or math.isinf(fb_new)):
                if abs(fb_new) < best_abs_fx:
                    best_x = b_new
                    best_fx = fb_new
                    best_abs_fx = abs(fb_new)
                
                if fb_new * best_fx < 0:
                    a = x
                    b = b_new
                    fa = best_fx
                    fb = fb_new
                    bracket_found = True
                    break
            
            step_size *= 2.0
            
            if step_size > 1e6:
                break

    if bracket_found:
        # Use Illinois Method
        # Illinois is a variant of Brent's method that replaces the secant step with a weighted bisection
        # when the secant step is too large, making it very robust for ill-conditioned problems.
        
        # Ensure fa and fb correspond to a and b
        # We have a, b, fa, fb such that fa * fb < 0
        
        # Illinois Algorithm:
        # 1. Evaluate f at a and b.
        # 2. If |f(a)| < |f(b)|, swap a and b (and fa, fb) so that |fa| > |fb|.
        #    Actually, standard Illinois keeps the point with smaller |f| as the "current" best guess?
        #    Let's stick to: c is the new point.
        #    We maintain a bracket [a, b] with f(a)f(b) < 0.
        #    We also keep track of the point with the smallest |f| to return if we run out of time.
        
        # Let's ensure a is the "bad" end (larger |f|) and b is the "good" end (smaller |f|)?
        # No, let's just follow the standard Illinois logic:
        # Let x1, x2 be the bracket ends.
        # x0 is the point with smaller |f|.
        # Calculate x3 = x1 - f(x1)*(x2-x1)/(f(x2)-f(x1)) (Secant)
        # If x3 is outside [x1, x2] or |f(x3)| > |f(x0)|, then x3 = (x1+x2)/2 (Bisection)
        # Actually, Illinois specifically uses:
        # If the secant point is not between x1 and x2, use bisection.
        # Also, if |f(x3)| > |f(x0)|, use bisection.
        # Then, replace the end with larger |f| with x3.
        
        x1 = a
        x2 = b
        f1 = fa
        f2 = fb
        
        # Ensure f1 is the one with larger magnitude? Not strictly necessary for correctness, 
        # but helps heuristic. Let's just start.
        
        for _ in range(500):
            if timed_out():
                break
            
            # Check if we converged
            if abs(f1) <= 1e-8:
                best_x = x1
                best_fx = f1
                best_abs_fx = abs(f1)
                return best_x
            if abs(f2) <= 1e-8:
                best_x = x2
                best_fx = f2
                best_abs_fx = abs(f2)
                return best_x
                
            # Determine which end to replace.
            # We want to keep the end with smaller |f| fixed? No, we replace the end with larger |f|.
            if abs(f1) > abs(f2):
                # x1 is the "bad" end, x2 is the "good" end
                x_bad = x1
                f_bad = f1
                x_good = x2
                f_good = f2
            else:
                x_bad = x2
                f_bad = f2
                x_good = x1
                f_good = f1
            
            # Secant estimate
            denom = f_good - f_bad
            if denom == 0:
                # Degenerate, use bisection
                x_new = (x_bad + x_good) / 2.0
            else:
                x_new = x_bad - f_bad * (x_good - x_bad) / denom
            
            # Check if x_new is inside the bracket
            if (x_new - x_bad) * (x_new - x_good) > 0:
                # Outside bracket, use bisection
                x_new = (x_bad + x_good) / 2.0
            
            # Evaluate
            f_new = f(x_new)
            eval_count += 1
            if timed_out():
                break
                
            if math.isnan(f_new) or math.isinf(f_new):
                # If evaluation fails, shrink bracket to bisection
                x_new = (x_bad + x_good) / 2.0
                f_new = f(x_new)
                eval_count += 1
                if timed_out():
                    break
                if math.isnan(f_new) or math.isinf(f_new):
                    break

            # Update best
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
                
            # Replace the "bad" end
            if abs(f_bad) > abs(f_new):
                # f_new is better than f_bad, so replace x_bad with x_new
                if abs(f1) > abs(f2):
                    # x1 was bad
                    x1 = x_new
                    f1 = f_new
                else:
                    # x2 was bad
                    x2 = x_new
                    f2 = f_new
            else:
                # f_bad is still better (or equal) than f_new? 
                # This implies the root is not in the direction we stepped? 
                # Or we just didn't improve.
                # In Illinois, if the new point is not better than the "good" point?
                # Actually, if |f_new| > |f_good|, we should have used bisection?
                # Let's check the condition: If |f_new| > |f_good|, replace the OTHER end?
                # No, standard Illinois:
                # "If |f(x3)| > |f(x0)|, then x3 is replaced by the midpoint."
                # Here x0 is the good point.
                # So if |f_new| > |f_good|, we should have used midpoint.
                # Let's re-evaluate:
                if abs(f_new) > abs(f_good):
                    # We should have used midpoint. Let's force it for next iteration?
                    # Or just accept it and let the bracket shrink?
                    # If we accept x_new, the bracket becomes [x_good, x_new] or [x_new, x_bad]?
                    # Since f_bad and f_good have opposite signs, and f_new has same sign as f_bad (because |f_new| > |f_good| doesn't guarantee sign, but usually in secant it does if convex).
                    # If f_new has same sign as f_bad, the new bracket is [x_new, x_good].
                    # If f_new has same sign as f_good, the new bracket is [x_bad, x_new].
                    # Let's just check signs.
                    if f_new * f_bad > 0:
                        # Same sign as bad, so new bad end is x_new
                        if abs(f1) > abs(f2):
                            x1 = x_new
                            f1 = f_new
                        else:
                            x2 = x_new
                            f2 = f_new
                    else:
                        # Same sign as good? That means we crossed the root?
                        # Then x_new is the new good end?
                        # This is getting complicated.
                        # Let's stick to the simple rule: Replace the end with larger |f|.
                        # If |f_bad| > |f_new|, replace bad.
                        # If |f_good| > |f_new|, replace good.
                        # This maintains a bracket.
                        if abs(f1) > abs(f2):
                            # x1 was bad. If f_new is better than f2 (good)?
                            # Then replace x1 with x_new? No, if f_new is better than f2, 
                            # then x_new is closer to root than x2.
                            # But we need to maintain sign change.
                            # f1 * f2 < 0.
                            # If f_new * f1 < 0, then [x_new, x2] is a bracket.
                            # If f_new * f2 < 0, then [x1, x_new] is a bracket.
                            
                            pass # Logic handled by sign check below

                    # Simpler robust approach:
                    # Check sign of f_new.
                    if f_new * f1 < 0:
                        # x_new and x1 have opposite signs. Bracket is [x_new, x1]
                        x2 = x_new
                        f2 = f_new
                    elif f_new * f2 < 0:
                        # x_new and x2 have opposite signs. Bracket is [x2, x_new]
                        x1 = x_new
                        f1 = f_new
                    else:
                        # No sign change? This should not happen if f is continuous and we started with a bracket.
                        # It can happen with numerical noise.
                        # Shrink to bisection.
                        x_mid = (x1 + x2) / 2.0
                        f_mid = f(x_mid)
                        eval_count += 1
                        if timed_out(): break
                        if f_mid * f1 < 0:
                            x2 = x_mid
                            f2 = f_mid
                        else:
                            x1 = x_mid
                            f1 = f_mid

            # Convergence check based on bracket width
            if abs(x1 - x2) < 1e-15:
                # Return the one with smaller |f|
                if abs(f1) < abs(f2):
                    return x1
                else:
                    return x2

    return best_x
