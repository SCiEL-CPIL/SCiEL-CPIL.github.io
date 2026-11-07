import numpy as np
from scipy.integrate._ivp.rk import DOP853

def integrate(f, t_span, y0):
    t0, t1 = t_span
    h = 0.002
    n_steps = int(round((t1 - t0) / h))
    # Ensure we cover the span exactly?
    # If (t1-t0) is not multiple of h, adjust last step?
    # For fixed step, usually we just do n_steps and stop.
    # The task asks for state at end of t_span.
    # If we don't hit t1 exactly, we might have a small error.
    # Better to adjust h slightly?
    # h = (t1 - t0) / n_steps
    # Let's pick n_steps = 1000.
    # h = (t1 - t0) / 1000
    
    n_steps = 1000
    h = (t1 - t0) / n_steps
    
    # Get coefficients
    A = DOP853.A
    c = DOP853.c
    b = DOP853.b
    n_stages = len(c)
    
    y = np.array(y0, dtype=float)
    t = t0
    
    for i in range(n_steps):
        t_i = t0 + i * h
        k = np.zeros((n_stages, len(y)))
        for s in range(n_stages):
            # y_s = y + h * sum(A[s, j] * k[j])
            y_s = y + h * np.dot(A[s, :s], k[:s]) # A[s, s] is 0 usually?
            # DOP853 A is lower triangular.
            # np.dot(A[s, :s], k[:s]) works if A[s,s] is 0.
            # Let's be safe:
            y_s = y.copy()
            for j in range(s):
                y_s += A[s, j] * h * k[j]
            k[s] = f(t_i + c[s] * h, y_s)
        
        y += h * np.dot(b, k)
        
    return y
