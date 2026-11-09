def find_root(f, x0):
    MAX_CALLS = 480
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        calls[0] += 1
        return f(x)
    
    x = float(x0)
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Bracket
    step = 1e-2 * max(1.0, abs(x))
    a, b = x - step, x + step
    fa = feval(a)
    fb = feval(b)
    
    if fa is None or fb is None:
        return x
        
    if abs(fa) <= 1e-8: return a
    if abs(fb) <= 1e-8: return b
    
    if fa * fb > 0:
        expanded = 0
        while fa * fb > 0 and expanded < 50 and calls[0] < MAX_CALLS - 10:
            step *= 2.0
            a_new, b_new = x - step, x + step
            fa_new = feval(a_new)
            fb_new = feval(b_new)
            if fa_new is None or fb_new is None:
                break
            if abs(fa_new) <= 1e-8: return a_new
            if abs(fb_new) <= 1e-8: return b_new
            if fa_new * fb_new < 0:
                a, b, fa, fb = a_new, b_new, fa_new, fb_new
                break
            a, b = a_new, b_new
            fa, fb = fa_new, fb_new
            expanded += 1
            
    if fa * fb > 0:
        # Fallback to Secant if no bracket
        c = x + step
        fc = feval(c)
        if fc is None: return x
        x1, x2 = x, c
        f1, f2 = fx, fc
        for _ in range(100):
            if calls[0] >= MAX_CALLS: break
            if f1 == f2: break
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            f3 = feval(x3)
            if f3 is None: break
            if abs(f3) <= 1e-8: return x3
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        return x2

    # Phase 2: Brent's Method
    # a, b are the bracket with fa*fb < 0
    
    # Initialize
    c = a
    d = a
    e = 0.0
    fc = fa
    
    # Ensure |fb| < |fa|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        c, d = b, a
        e = b - a
        fc = fb
        
    tol = 1e-8
    xtol = 1e-15
    
    for _ in range(200):
        if calls[0] >= MAX_CALLS:
            break
            
        if abs(fb) <= tol:
            return b
            
        if abs(b - a) < xtol * max(1.0, abs(a), abs(b)):
            return b
            
        # 1. If f(a) * f(c) < 0
        if fa * fc < 0:
            b, a = a, b
            fb, fa = fa, fb
            c, d = b, a
            e = b - a
            fc = fb
            
        # 2. If |c| < |b| (i.e., |fc| < |fb|)
        if abs(fc) < abs(fb):
            b, a = a, c
            fb, fa = fa, fc
            c, d = b, a
            e = b - a
            fc = fb
            
        # 3. m = e; e = d; d = b
        m = e
        e = d
        d = b
        
        # 4. Check convergence
        if abs(e) < xtol * max(1.0, abs(a), abs(b)) or abs(fb) < tol:
            return b
            
        # 5. Try IQI or Secant
        # S = f(b)/f(a) if |f(a)| < |f(b)| else 1
        if abs(fa) < abs(fb):
            S = fb / fa
        else:
            S = 1.0
            
        # If |e| >= tol and |S - 1| > tol
        if abs(e) >= xtol * max(1.0, abs(a), abs(b)) and abs(S - 1.0) > xtol:
            # Try IQI
            # p = c - (f(c) * ( (b-c)*fa/(fb-fa) ) / ( (a-c)*fb/(fb-fc) )) ?
            # Formula: p = b - (b - a) * ( f(b)/(f(b)-f(a)) ) * ( (b - c) / (a - c) ) ?
            # Standard IQI:
            # p = c - f(c) * ( (b-c)*fa/(fb-fa) ) / ( (a-c)*fb/(fb-fc) )
            # Let's use the form from Numerical Recipes:
            # p = 2*m*S - (b+c)*S - (b-a)*S ... no.
            
            # Simpler IQI formula:
            # p = b - (b - a) * ( f(b) / (f(b) - f(a)) ) * ( (b - c) / (a - c) )
            # This is actually Secant if c=a? No.
            
            # Let's use the standard formula:
            # p = c - f(c) * ( (b-c)*fa/(fb-fa) ) / ( (a-c)*fb/(fb-fc) )
            # Denominator check
            den1 = (fb - fa)
            den2 = (fb - fc)
            
            if den1 != 0 and den2 != 0 and (a - c) != 0:
                p = c - fc * ( (b - c) * fa / den1 ) / ( (a - c) * fb / den2 )
                
                # Check if p is within [a, b] and not too close to a
                # Condition: p > (3*a + b)/4 and p < (a + 3*b)/4
                if p > (3.0*a + b) / 4.0 and p < (a + 3.0*b) / 4.0:
                    # Accept IQI
                    pass
                else:
                    # Fall back to Secant
                    if den1 != 0:
                        p = b - fb * (b - a) / den1
                    else:
                        p = (a + b) / 2.0
                        
            else:
                # Fall back to Secant
                if den1 != 0:
                    p = b - fb * (b - a) / den1
                else:
                    p = (a + b) / 2.0
        else:
            # Bisection
            p = (a + b) / 2.0
            
        # Check if p is valid
        if p <= min(a, b) or p >= max(a, b):
            # Bisection
            p = (a + b) / 2.0
        elif abs(p - b) < xtol * max(1.0, abs(b), abs(p)):
            # Too close to b
            p = (a + b) / 2.0
            
        # Evaluate f(p)
        fp = feval(p)
        if fp is None:
            break
            
        if abs(fp) <= tol:
            return p
            
        # Update bracket
        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp
            
        # Update c, d, e
        # c is the previous a? No, in Brent's, c is updated to the old b if b is swapped?
        # Standard:
        # if f(a)*f(c) < 0: ...
        # if |c| < |b|: ...
        # m = e; e = d; d = b
        # This logic is embedded in the loop start.
        
        # We need to update c for the next iteration's IQI.
        # c should be the point that was 'a' in the previous iteration?
        # Actually, the standard algorithm updates c to be the old b if b is swapped with a?
        # No, c is simply the previous a.
        # But in the loop, we don't explicitly store "previous a" as c unless we swap.
        # Let's re-read the standard algorithm.
        # 
        # 1. if f(a)*f(c) < 0: b=a, a=c, c=b, d=b, e=b-a
        # 2. if |c| < |b|: b=a, a=c, c=b, d=b, e=b-a
        # 3. m=e, e=d, d=b
        # 
        # This implies c is maintained.
        # In my loop, I am not updating c correctly in the "else" branch (no swap).
        # If no swap happens, c remains the same?
        # Yes, c is only updated if a or b is swapped in steps 1 or 2.
        # But if we just move b to p (without swapping a and b), c remains the old c.
        # Is that correct?
        # Yes, c is the "previous a" only in the sense of the last time the bracket was updated via swap?
        # Actually, c is just a point that has been evaluated.
        # For IQI, we need 3 points: a, b, c.
        # If we don't swap, a and c might be the same?
        # Initially c=a.
        # If we don't swap, a stays a. c stays a.
        # Then IQI fails (division by zero or duplicate points).
        # We need to ensure c is distinct from a and b.
        # 
        # Correction: In Brent's Method, c is the point that was b in the previous iteration?
        # No, c is the previous a.
        # If we don't swap, a is the same. So c = a.
        # Then IQI is not possible. We use Secant or Bisection.
        # This is fine.
        
        # However, we need to update c to be the new a if a changes?
        # In the standard algorithm, c is updated ONLY in the swap steps.
        # If a changes without a swap (i.e., b is replaced by p, and a stays), then c is not updated.
        # This means c might be equal to a.
        # This is acceptable; IQI will just fail and we'll use Secant/Bisection.
        
        # But wait, if we do swap (step 1 or 2), c becomes the old b.
        # So c is always a point that was previously b or a.
        
        # My implementation above does NOT update c in the non-swap case.
        # This is correct per the standard algorithm.
        
        # One issue: if we don't swap, and a is unchanged, c is unchanged.
        # If c == a, IQI fails.
        
        # Let's verify the loop structure again.
        # The standard algorithm updates c, d, e at the BEGINNING of the loop, based on previous state.
        # My code does the swaps at the beginning.
        # Then it calculates p.
        # Then it updates a or b.
        # It does NOT update c, d, e at the end.
        # This is correct. c, d, e are updated at the start of the NEXT iteration.
        
    return b
