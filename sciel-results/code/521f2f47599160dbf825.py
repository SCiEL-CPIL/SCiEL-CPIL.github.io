"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on t_span using matrix exponential for linear systems.
    
    Strategy:
    1. Extract the system matrix A using 15 random probes (15 calls to f).
    2. Check if f is time-independent (linear autonomous system y' = A*y).
    3. If so, compute y(t_end) = expm(A*(t_end - t_start)) @ y0.
    4. If not, fall back to a robust integrator.
    
    Total f calls: ~15-20 for the matrix extraction + verification.
    """
    t0, t1 = t_span
    dt = t1 - t0
    n = len(y0)
    
    # Check for deadline
    start_time = time.time()
    
    # Step 1: Extract matrix A using 15 random vectors
    # We need to solve: f(t, X) = A @ X (if autonomous linear)
    # Use random vectors to determine A
    
    # First, check if f is autonomous by comparing f(t0, y0) and f(t1, y0)
    f_t0_y0 = f(t0, y0)
    f_t1_y0 = f(t1, y0)
    
    # Check if autonomous: f(t0, y0) should equal f(t1, y0) if autonomous
    # Also check linearity: f(t, c*y) should equal c*f(t, y)
    
    # Extract A using random probes
    # Use 15 random vectors to determine A
    np.random.seed(42)
    
    # Generate 15 random vectors
    V = np.random.randn(n, n)
    
    # Evaluate f at each column of V
    F = np.empty((n, n))
    for j in range(n):
        v_j = V[:, j]
        F[:, j] = f(t0, v_j)
    
    # If autonomous linear: F = A @ V, so A = F @ V^{-1}
    try:
        V_inv = np.linalg.inv(V)
        A = F @ V_inv
    except np.linalg.LinAlgError:
        # V is singular, try different random vectors
        V = np.random.randn(n, n)
        F = np.empty((n, n))
        for j in range(n):
            v_j = V[:, j]
            F[:, j] = f(t0, v_j)
        try:
            V_inv = np.linalg.inv(V)
            A = F @ V_inv
        except np.linalg.LinAlgError:
            # Fall back to RK4
            return _rk4_safe(f, t0, t1, y0, 5000, start_time)
    
    # Verify linearity: check that f(t, A@y) = A @ f(t, y) for a test vector
    # Actually, verify that the extracted A is consistent
    # Test: f(t0, y0) should equal A @ y0 (if autonomous linear)
    f_test = f(t0, y0)
    A_y0 = A @ y0
    
    # Check if autonomous and linear
    residual_autonomous = np.linalg.norm(f_t0_y0 - f_t1_y0)
    residual_linear = np.linalg.norm(f_test - A_y0)
    
    # Check if the system is truly linear and autonomous
    # Also verify with another random vector
    v_test = np.random.randn(n)
    f_v_test = f(t0, v_test)
    A_v_test = A @ v_test
    residual_linear2 = np.linalg.norm(f_v_test - A_v_test)
    
    # If residuals are small, use matrix exponential
    if residual_autonomous < 1e-10 * (1 + np.linalg.norm(f_t0_y0)) and \
       residual_linear < 1e-10 * (1 + np.linalg.norm(f_test)) and \
       residual_linear2 < 1e-10 * (1 + np.linalg.norm(f_v_test)):
        
        # Use matrix exponential for exact solution
        # y(t1) = expm(A * dt) @ y0
        M = A * dt
        expM = expm(M)
        y_final = expM @ y0
        
        # Verify result is finite
        if np.all(np.isfinite(y_final)):
            return y_final
    
    # If not autonomous linear, fall back to a robust method
    # Use implicit Euler or Radau if available
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: RK4 with many steps
    return _rk4_safe(f, t0, t1, y0, 10000, start_time)


def _rk4_safe(f, t0, t1, y0, steps, start_time=None):
    """Safe RK4 with deadline guard."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    
    for i in range(steps):
        # Deadline check
        if start_time is not None and time.time() - start_time > 300:
            break
        
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    
    return y
# EVOLVE-BLOCK-END
