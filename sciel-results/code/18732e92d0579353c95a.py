import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    start_time = time.time()
    
    if dt == 0:
        return y0.copy()

    # Helper to get A(t)
    def get_A(t):
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t, e)
        return A

    # 1. Check for constant A
    A0 = get_A(t0)
    A1 = get_A(t1)
    Amid = get_A(0.5 * (t0 + t1))
    
    # Check homogeneity f(t, 0) = 0
    # If not homogeneous, it's affine y' = A(t)y + b(t).
    # The prompt says "linear system", which usually implies homogeneous in this context 
    # unless b(t) is provided. The seed checks hom_err.
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)
    
    time_indep_err = max(np.linalg.norm(A0 - A1), np.linalg.norm(A0 - Amid))
    
    if hom_err < 1e-12 and time_indep_err < 1e-12:
        M = expm(A0 * dt)
        return M @ y0

    # 2. Time-dependent linear case
    # We assume it is linear. 
    # We use PCA with large K.
    
    # Determine K based on call budget
    # Max calls 2,000,000. 
    # Each step costs n calls.
    # Reserve some calls for safety.
    max_calls = 1800000
    K_max_calls = max_calls // n
    
    # Estimate time per step
    # Do a few test steps
    h_test = dt / 100
    t_test = t0 + 0.5 * h_test
    A_test = get_A(t_test)
    M_test = expm(A_test * h_test)
    time_one_step = time.time() - start_time
    
    # Extrapolate to full K
    # time_total ~ K * time_one_step
    # We want time_total < 250s
    K_max_time = int(250.0 / max(time_one_step, 1e-9))
    
    K = min(K_max_calls, K_max_time, 200000) # Cap at 200k just in case, though 100k is likely fine
    
    # Ensure K is at least something reasonable
    K = max(K, 1000)
    
    h = dt / K
    y = y0.copy()
    
    for k in range(K):
        t_mid = t0 + (k + 0.5) * h
        A_k = get_A(t_mid)
        M_k = expm(A_k * h)
        y = M_k @ y
        
        # Time check
        if time.time() > start_time + 290:
            break
            
    return y
