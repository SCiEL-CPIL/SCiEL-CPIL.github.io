import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    t = t1 - t0
    y = np.array(y0, dtype=float)
    
    # Taylor series expansion of e^(At) y0
    # y(t) = sum_{k=0}^inf (t^k / k!) A^k y0
    # v_k = A^k y0
    # v_0 = y0
    # v_k = f(0, v_{k-1})
    
    v = y.copy()
    term = y.copy()
    y_end = y.copy()
    
    # Pre-compute powers of t / k!
    # We will update term recursively: term_k = term_{k-1} * (t/k) * A
    # Actually, v_k = A v_{k-1}.
    # term_k = (t^k / k!) v_k.
    # term_0 = y0.
    # term_k = term_{k-1} * (t/k) * A ? No.
    # term_k = (t^k/k!) A^k y0
    # term_{k-1} = (t^{k-1}/(k-1)!) A^{k-1} y0
    # term_k = term_{k-1} * (t/k) * A
    # So we need to apply A to term_{k-1} * (t/k).
    # Let w = term_{k-1} * (t/k). Then v_k = f(0, w) * k! ? No.
    # v_k = A v_{k-1}.
    # term_k = (t^k/k!) v_k = (t/k) (t^{k-1}/(k-1)!) v_k = (t/k) (t^{k-1}/(k-1)!) A v_{k-1}.
    # This doesn't simplify directly to a single f call on the previous term.
    # We need to keep track of v_k separately or scale it.
    
    # Let's just keep v and term.
    # v_0 = y0
    # term_0 = y0
    # For k=1:
    # v_1 = f(0, v_0)
    # term_1 = (t/1) * v_1
    # For k=2:
    # v_2 = f(0, v_1)
    # term_2 = (t/2) * term_1 ? No.
    # term_2 = (t^2/2) v_2.
    # term_1 = t v_1.
    # term_2 = (t/2) t v_2 = (t/2) f(0, v_1) * t ? No.
    # term_2 = (t^2/2) A^2 y0.
    # term_1 = t A y0.
    # term_2 = (t/2) * (t A y0) * A ? No, dimensions don't match for simple multiplication.
    # term_2 = (t/2) * (term_1 / t) * A ?
    # Let's just compute v_k and term_k explicitly.
    
    v = y.copy()
    term = y.copy()
    y_end = y.copy()
    
    max_terms = 100
    tol = 1e-15
    
    for k in range(1, max_terms):
        # Compute v_k = A v_{k-1}
        v = f(0, v)
        # Compute term_k = (t^k / k!) v_k
        # term_{k-1} = (t^{k-1} / (k-1)!) v_{k-1}
        # term_k = (t/k) * (t^{k-1} / (k-1)!) v_k
        # We don't have (t^{k-1}/(k-1)!) v_k directly.
        # But we know v_k = A v_{k-1}.
        # So we just calculate the scalar factor.
        # Let scalar_k = t^k / k!
        # We can update scalar recursively: scalar_k = scalar_{k-1} * (t/k)
        # But we need v_k.
        # So:
        # v_k = f(0, v_{k-1})
        # term_k = scalar_k * v_k
        
        # To avoid storing all scalars, we can just update a running scalar.
        # But we need v_k.
        # So the loop is:
        # v = f(0, v)  # this is v_k
        # term = (t/k) * (term / (t/(k-1))) ? No.
        
        # Let's restart the logic inside the loop.
        pass

    # Correct Logic:
    v = y.copy()
    term = y.copy()
    y_end = y.copy()
    
    # scalar_k = t^k / k!
    # We can maintain a variable `coef` which is t^k / k!
    # coef_0 = 1
    # coef_k = coef_{k-1} * t / k
    
    coef = 1.0
    for k in range(1, max_terms):
        # Update v to v_k = A v_{k-1}
        v = f(0, v)
        # Update coef to coef_k
        coef *= t / k
        # term_k = coef_k * v_k
        term = coef * v
        y_end += term
        
        # Check convergence
        if np.linalg.norm(term) < tol * max(1.0, np.linalg.norm(y_end)):
            break
            
    return y_end
