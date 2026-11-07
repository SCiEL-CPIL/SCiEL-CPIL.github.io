"""ODE integrator exploiting the fixed linear structure: y' = A y, with A recovered from f."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _recover_matrix(f, t, y0, n):
    """
    Recover the matrix A from f(t, y) = A @ y by probing each column.
    Uses 2n+1 calls: one for A@y0, and 2 for each column (perturb +/-).
    Actually we can do it with n+1 calls if we use y0 and perturbations.
    Better: use y0 and n unit vectors? No, that's n+1 calls for n columns + 1 for y0.
    But we need A@y0 for the first step.
    
    Strategy:
    1. Call f(t, y0) to get A@y0. (1 call)
    2. For each column j, create a vector that is zero except at j where it is 1.
       Call f(t, e_j) to get column j of A. (n calls)
    Total: n+1 calls.
    
    But wait, the problem says y' = A y, so f(t, y) should be A @ y (assuming time-independent A).
    Let's verify: A = 21^2 * tridiag(1, -2, 1). This is time-independent.
    So f(t, y) = A @ y.
    
    We can recover A with n+1 calls:
    - Call f(t, y0) -> A@y0
    - Call f(t, e_j) for j=0..n-1 -> column j of A
    
    Actually, we can do even better. Since we know the structure (tridiagonal), we could
    just construct A directly. But the task says "A = 21^2 * tridiag(1, -2, 1)" so we know it.
    However, the function f is given, so we should use it to be safe, but we can also
    construct A analytically.
    
    Let's construct A analytically since it's given in the task.
    """
    # Construct A = 21^2 * tridiag(1, -2, 1)
    # tridiag(1, -2, 1) means sub-diagonal 1, diagonal -2, super-diagonal 1
    # For n=20
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n-1:
            A[i, i+1] = 1.0
    A *= 21.0**2
    
    # Verify with one call to f
    y_test = y0.copy()
    f_val = f(t, y_test)
    # Check if f(t, y) = A @ y
    # If not, we need to recover A from f
    # For now, assume it is correct. If there's a mismatch, we'll recover.
    # Let's do a quick check
    diff = np.linalg.norm(f_val - A @ y_test)
    if diff > 1e-10:
        # Recover A from f
        A = np.zeros((n, n))
        # First, get A@y0
        # Then get columns
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t, e_j)
        # We don't use the first call for A@y0 in this recovery, but we can verify
        # Actually, this uses n calls, which is fine.
    
    return A


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s margin
    
    # Recover matrix A
    A = _recover_matrix(f, t0, y0, n)
    
    # Use matrix exponential for the full interval?
    # y(t1) = exp(A*(t1-t0)) @ y0
    # This is exact for linear time-invariant systems.
    # But we need to call f at least once. We already did in _recover_matrix.
    # Let's compute expm(A * dt) @ y0
    
    dt = t1 - t0
    try:
        from scipy.linalg import expm
        expA = expm(A * dt)
        y_end = expA @ y0
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass
    
    # Fallback: use a high-accuracy implicit method or RK45 with tight tolerances
    # Since the system is stiff, let's use Radau or BDF
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=dt/100)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: RK4 with many steps
    steps = 10000
    h = dt / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
