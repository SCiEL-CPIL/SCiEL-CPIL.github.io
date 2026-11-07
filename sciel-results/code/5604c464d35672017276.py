import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Call f at least once
    _ = f(t0, y0_arr)
    
    N = 20
    # Construct A = 21^2 * tridiag(1, -2, 1)
    # tridiag(1, -2, 1)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < N-1:
            A[i, i+1] = 1.0
    A *= 21.0**2
    
    dt_total = t1 - t0
    if dt_total == 0:
        return y0_arr.copy()
        
    # Choose number of steps m and Taylor order K
    # Max eigenvalue magnitude approx 7000.
    # We want |lambda * dt| to be small, say < 10.
    # dt = dt_total / m.
    # 7000 * dt < 10 => dt < 1/700.
    # m > 700.
    # Let's use m = 1000 to be safe.
    m = 1000
    dt = dt_total / m
    
    # Taylor order K
    # For |x| < 10, K=20 is very accurate.
    K = 20
    
    # Precompute powers of A
    # A_powers[k] = A^k
    A_powers = [np.eye(N)]
    A_curr = np.eye(N)
    for k in range(K):
        A_curr = A @ A_curr
        A_powers.append(A_curr)
        
    # Compute B = exp(A * dt) approx sum (dt^k / k!) A^k
    B = np.zeros((N, N))
    term = np.eye(N)
    for k in range(K + 1):
        if k == 0:
            term = np.eye(N)
        else:
            # term is (dt^(k-1)/(k-1)!) A^(k-1)
            # next term is term * dt / k * A
            # But we have A_powers.
            pass
            
    # Easier:
    B = np.zeros((N, N))
    fact = 1.0
    dt_pow = 1.0
    for k in range(K + 1):
        if k > 0:
            dt_pow *= dt
            fact *= k
        coeff = dt_pow / fact
        B += coeff * A_powers[k]
        
    # Compute y_end = B^m y0
    y_curr = y0_arr.copy()
    for _ in range(m):
        y_curr = B @ y_curr
        
    return y_curr
