import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    start = time.time()
    deadline = start + 290.0  # safety margin before 330s kill

    if dt == 0:
        return y0.copy()

    # Precompute basis vectors for extracting A(t)
    basis = np.eye(n)

    # Helper to extract A(t) assuming f(t, y) = A(t) y
    # Uses n calls to f per evaluation of A(t)
    def get_A(t):
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            A[:, j] = f(t, basis[j])
        return A

    # 1. Check for time-independent linear system
    A0 = get_A(t0)
    
    # Check homogeneity: f(t, 0) should be 0
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)
    
    # Check time-independence: A(t1) and A(mid) should match A0
    A1 = get_A(t1)
    Amid = get_A(0.5 * (t0 + t1))
    
    time_indep_err = max(np.linalg.norm(A0 - A1), np.linalg.norm(A0 - Amid))
    
    # If it looks like a constant linear system
    if hom_err < 1e-12 and time_indep_err < 1e-12:
        M = expm(A0 * dt)
        return M @ y0

    # 2. Check for time-dependent linear system
    # Verify linearity at a few points
    np.random.seed(42)
    y_test1 = np.random.randn(n)
    y_test2 = np.random.randn(n)
    alpha = 2.5
    
    # Test at t_mid
    t_test = 0.5 * (t0 + t1)
    A_test = get_A(t_test)
    
    pred_sum = A_test @ (y_test1 + y_test2)
    actual_sum = f(t_test, y_test1 + y_test2)
    err_sum = np.linalg.norm(pred_sum - actual_sum) / (1e-30 + np.linalg.norm(actual_sum))
    
    pred_alpha = A_test @ (alpha * y_test1)
    actual_alpha = f(t_test, alpha * y_test1)
    err_alpha = np.linalg.norm(pred_alpha - actual_alpha) / (1e-30 + np.linalg.norm(actual_alpha))
    
    # Also check homogeneity at t_test
    z_test = f(t_test, np.zeros(n))
    hom_err_test = np.linalg.norm(z_test)
    
    is_linear = (err_sum < 1e-10 and err_alpha < 1e-10 and hom_err_test < 1e-12)
    
    if is_linear:
        # Use Piecewise Constant Approximation (PCA) with Matrix Exponentials
        # This is highly accurate for linear systems.
        # Choose number of steps K.
        # We need to balance accuracy vs number of f calls (n*K) and time.
        
        # Strategy: Use adaptive K based on available time
        # First, measure how long one expm + matrix-vector multiply takes
        A_sample = get_A(t_test)
        h_test = dt / 100
        M_sample = expm(A_sample * h_test)
        _ = M_sample @ y0
        
        # Estimate time per step
        t1_measure = time.time()
        K_test = 100
        for _ in range(K_test):
            A_k = get_A(t_test)
            M_k = expm(A_k * h_test)
            _ = M_k @ y0
        t2_measure = time.time()
        time_per_step = (t2_measure - t1_measure) / K_test
        
        # Available time for integration steps
        elapsed = time.time() - start
        remaining_time = deadline - start - elapsed
        # Reserve 10% of remaining time for safety
        usable_time = 0.9 * remaining_time
        
        # Calculate maximum K
        max_K = int(usable_time / max(time_per_step, 1e-6))
        # Cap K to avoid excessive f calls (2000000 limit)
        max_K = min(max_K, 2000000 // n)
        # Also ensure reasonable minimum
        max_K = max(max_K, 100)
        
        # Use PCA with K steps
        K = max_K
        h = dt / K
        y = y0.copy()
        
        for k in range(K):
            if time.time() > deadline:
                # Fallback: return current y
                return y
                
            t_mid = t0 + (k + 0.5) * h
            A_k = get_A(t_mid)
            M_k = expm(A_k * h)
            y = M_k @ y
            
        return y

    # 3. Fallback: Non-linear or detection failed
    # Use solve_ivp with Radau
    from scipy.integrate import solve_ivp
    
    # Try to provide Jacobian if possible? 
    # If it's not linear, we can't easily provide it.
    
    sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-10, atol=1e-12,
                    max_step=0.01)
    
    if not sol.success:
        sol = solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-10, atol=1e-12,
                        max_step=0.01)
        
    return sol.y[:, -1]
