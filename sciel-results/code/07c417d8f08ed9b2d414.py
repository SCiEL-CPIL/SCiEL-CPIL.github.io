import numpy as np
import time
import scipy.linalg

def _arnoldi_exp(A_func, t, v0, m=None):
    """
    Compute exp(t*A) @ v0 using Arnoldi iteration (Krylov subspace method).
    
    Parameters:
    - A_func: function that applies A to a vector, i.e., A_func(x) = A @ x
    - t: scalar time step
    - v0: initial vector
    - m: dimension of Krylov subspace (default: min(n, 15))
    
    Returns:
    - approximation of exp(t*A) @ v0
    """
    n = v0.shape[0]
    if m is None:
        m = min(n, 15)
    
    # Normalize v0
    v0_norm = np.linalg.norm(v0)
    if v0_norm < 1e-15:
        return np.zeros_like(v0)
    
    v0_unit = v0 / v0_norm
    
    # Arnoldi iteration
    V = np.zeros((n, m))
    H = np.zeros((m, m))
    
    v = v0_unit.copy()
    V[:, 0] = v
    
    for j in range(m - 1):
        w = A_func(V[:, j])
        
        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        
        # Renormalize
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] > 1e-15:
            v = w / H[j+1, j]
            V[:, j+1] = v
        else:
            # Krylov subspace is invariant
            m_actual = j + 1
            break
    else:
        m_actual = m
    
    # Compute exp(t*H_m) @ e_1 where H_m is the m_actual x m_actual upper Hessenberg
    H_m = H[:m_actual, :m_actual].copy()
    H_m_exp = scipy.linalg.expm(t * H_m)
    
    # Result: v0_norm * V @ (H_m_exp @ e_1)
    e1 = np.zeros(m_actual)
    e1[0] = 1.0
    coeff = H_m_exp @ e1
    
    result = v0_norm * (V[:, :m_actual] @ coeff)
    
    return result

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    dt = t1 - t0

    if dt == 0.0:
        return y0.copy()

    start_time = time.time()
    deadline = start_time + 285.0

    # First, check if the system is homogeneous by evaluating f at zero
    b = np.asarray(f(t0, np.zeros(n)), dtype=float)
    b_norm = np.linalg.norm(b)
    
    # Check time remaining
    if time.time() > deadline:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]
    
    if b_norm < 1e-14:
        # Homogeneous system: y' = A y
        # Use Arnoldi to compute exp(dt*A) @ y0 without forming A
        def A_apply(x):
            return np.asarray(f(t0, x), dtype=float)
        
        y_end = _arnoldi_exp(A_apply, dt, y0, m=n)
    else:
        # Affine system: y' = A y + b
        # Use block matrix approach but with Krylov for the A part
        # For affine system, we can use: y(t) = exp(tA)y0 + integral_0^t exp((t-s)A) b ds
        # For constant b: y(t) = exp(tA)y0 + (exp(tA) - I) A^{-1} b
        # But this requires A^{-1} which we don't have.
        # Alternative: use the block matrix exponential
        # Block = [[A, b], [0, 1]]
        # We need to form A for this, so fall back to probing A
        
        # Probe A columns
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            col = np.asarray(f(t0, ei), dtype=float) - b
            A[:, i] = col
        
        if time.time() > deadline:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
            return sol.y[:, -1]
        
        # Use block matrix exponential
        block = np.zeros((n + 1, n + 1))
        block[:n, :n] = A
        block[:n, n] = b
        block[n, n] = 1.0
        
        block_exp = scipy.linalg.expm(block * dt)
        y_end = block_exp[:n, :n] @ y0 + block_exp[:n, n]

    # Quick verification: check if the system is truly autonomous
    if time.time() < deadline - 5.0:
        test_y = np.random.randn(n)
        f_t1 = np.asarray(f(t1, test_y), dtype=float)
        
        # For homogeneous: expected = A @ test_y, but we don't have A explicitly
        # For affine: expected = A @ test_y + b
        # We can check by comparing f(t0, test_y) with f(t1, test_y)
        f_t0 = np.asarray(f(t0, test_y), dtype=float)
        diff = np.linalg.norm(f_t1 - f_t0)
        scale = np.linalg.norm(f_t0) + 1.0
        if diff / scale > 1e-8:
            # System is time-varying; fall back to numerical integration
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]

    return y_end
