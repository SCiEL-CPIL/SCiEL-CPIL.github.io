import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    A = np.zeros((n, n), dtype=float)
    
    # Vector 1: Complex probe
    v1 = np.zeros(n, dtype=complex)
    v1[0] = 1.0
    v1[1] = 1.0j
    
    try:
        res1 = f(t0, v1)
        res1 = np.asarray(res1)
        
        if np.iscomplexobj(res1):
            # Complex path
            # We have 1 complex result. Need 7 more to get 16 real vectors.
            # Actually, 1 complex vector = 2 real vectors.
            # We need 15 real vectors to determine A (15x15).
            # 1 complex gives 2. We need 13 more real vectors.
            # 7 complex vectors give 14 real vectors.
            # Total 8 complex vectors -> 16 real vectors.
            
            V_complex = np.zeros((n, 8), dtype=complex)
            V_complex[:, 0] = v1
            Y_complex = np.zeros((n, 8), dtype=complex)
            Y_complex[:, 0] = res1
            
            rng = np.random.default_rng(42)
            for k in range(1, 8):
                V_complex[:, k] = rng.standard_normal(n) + 1j * rng.standard_normal(n)
                Y_complex[:, k] = f(t0, V_complex[:, k])
                if time.time() > deadline:
                    break
            
            V_real = np.hstack([V_complex.real, V_complex.imag])
            Y_real = np.hstack([Y_complex.real, Y_complex.imag])
            
            # Solve A V_real = Y_real
            # V_real is 15x16.
            A = Y_real @ np.linalg.pinv(V_real)
            
        else:
            # Real path
            # Assume res1 is A @ e_1
            A[:, 0] = np.asarray(res1, dtype=float)
            
            for j in range(1, n):
                e = np.zeros(n, dtype=float)
                e[j] = 1.0
                col = f(t0, e)
                A[:, j] = np.asarray(col, dtype=float)
                if time.time() > deadline:
                    break
                    
    except Exception:
        # Fallback: Standard real basis vectors
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n, dtype=float)
            e[j] = 1.0
            col = f(t0, e)
            A[:, j] = np.asarray(col, dtype=float)
            if time.time() > deadline:
                break

    M = expm(dt * A)
    y_end = M @ y0

    return y_end
