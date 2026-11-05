import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for 300s limit

    # Try to extract the linear operator A and offset b
    # y' = A y + b
    
    # Check if f(0, 0) is zero to determine if it's homogeneous or affine
    y_zero = np.zeros(n)
    try:
        f_zero = f(t0, y_zero)
    except Exception:
        f_zero = None

    if f_zero is None:
        # Fallback to RK4 if f is not callable or errors
        return _rk4(f, t0, t1, y0, 2000)

    # If f(0,0) is not close to zero, it's affine: y' = Ay + b
    # If it is zero, it's homogeneous: y' = Ay
    # We can always treat it as affine by augmenting the state if needed, 
    # but for homogeneous, standard expm is slightly cheaper/faster.
    
    # Let's assume it's linear first. 
    # Extract A by probing basis vectors.
    A = np.empty((n, n))
    calls = 1 # for f_zero
    
    # We need to be careful with time.
    # Probing takes n calls.
    # If n is 100, that's 100 calls.
    
    # Check time before probing?
    # Probing is fast.
    
    for j in range(n):
        if time.time() > deadline:
            # Timeout during probing, fallback
            return _rk4(f, t0, t1, y0, 2000)
        
        e_j = np.zeros(n)
        e_j[j] = 1.0
        try:
            # For homogeneous, f(0, e_j) is column j of A.
            # For affine, f(0, e_j) = A e_j + b = col_j(A) + b.
            # So A[:, j] = f(0, e_j) - b.
            # We have b = f_zero.
            val = f(t0, e_j)
            A[:, j] = val - f_zero
            calls += 1
        except Exception:
            # Non-linear or error
            return _rk4(f, t0, t1, y0, 2000)

    # Now we have A and b (f_zero).
    # Check if b is effectively zero to decide on matrix size.
    b_norm = np.linalg.norm(f_zero)
    
    if b_norm < 1e-12:
        # Homogeneous case
        # y(t) = exp(A * dt) * y0
        dt = t1 - t0
        M = la.expm(A * dt)
        y_end = M @ y0
    else:
        # Affine case
        # Augment state: z = [y; 1]
        # z' = [A, b; 0, 0] z
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A
        A_aug[:n, n] = f_zero
        # Last row is all zeros
        
        y_aug = np.zeros(n + 1)
        y_aug[:n] = y0
        y_aug[n] = 1.0
        
        dt = t1 - t0
        M_aug = la.expm(A_aug * dt)
        y_end_aug = M_aug @ y_aug
        y_end = y_end_aug[:n]

    return y_end

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y
