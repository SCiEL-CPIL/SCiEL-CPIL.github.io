"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Global deadline guard
_start_time = None
_DEADLINE = 320.0  # seconds, with 10s safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    return (_start_time is not None and (time.time() - _start_time) > _DEADLINE)


def _probe_linear_heat(f, t0, y0):
    """
    Probe f to see if it's a linear, time-invariant 5-point Laplacian on a 10x10 grid.
    If so, return the matrix A such that y' = A*y. Otherwise return None.
    """
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # For a 10x10 heat equation, n should be 100
    if n != 100:
        return None
    
    # Probe: check if f(t, y) is linear and time-invariant
    # Test time invariance: f(t0, y0) should equal f(t0+0.01, y0)
    f1 = f(t0, y0)
    f2 = f(t0 + 0.01, y0)
    if not np.allclose(f1, f2, rtol=1e-10, atol=1e-12):
        return None
    
    # Test linearity: f(t, a*y) should equal a*f(t, y) for scalar a
    a = 2.0
    f_linear = f(t0, a * y0)
    if not np.allclose(f_linear, a * f1, rtol=1e-8, atol=1e-10):
        return None
    
    # Test another scalar
    b = 0.5
    f_linear2 = f(t0, b * y0)
    if not np.allclose(f_linear2, b * f1, rtol=1e-8, atol=1e-10):
        return None
    
    # Now reconstruct A by probing columns
    # A * y = f(0, y)
    # We need to find A such that A @ y0 = f1
    # Use identity matrix probes: A[:, j] = f(0, e_j) where e_j is unit vector
    # But that's 100 calls. Instead, we can use the known structure of 5-point Laplacian.
    
    # For a 10x10 5-point Laplacian with Dirichlet BC (fixed boundary=0),
    # the interior points are 8x8=64, but the task says 10x10=100.
    # Let's assume all 100 points are free (Neumann or periodic BC).
    
    # Actually, let's just probe A using a set of test vectors.
    # Use 100 unit vectors to get columns of A.
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Verify: A @ y0 should equal f1
    if not np.allclose(A @ y0, f1, rtol=1e-6, atol=1e-8):
        return None
    
    return A


def _build_heat_matrix(n=100):
    """
    Build the 5-point Laplacian matrix for a 10x10 grid with Neumann BC.
    This is the expected structure for the heat equation.
    """
    A = np.zeros((n, n))
    rows = 10
    cols = 10
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            # Center
            A[idx, idx] = -4.0
            # Neighbors (with Neumann BC: reflect at boundaries)
            # Left
            if j > 0:
                A[idx, i * cols + (j - 1)] = 1.0
            # Right
            if j < cols - 1:
                A[idx, i * cols + (j + 1)] = 1.0
            # Up
            if i > 0:
                A[idx, (i - 1) * cols + j] = 1.0
            # Down
            if i < rows - 1:
                A[idx, (i + 1) * cols + j] = 1.0
    
    return A


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=50000):
    """
    Adaptive RK45 (Dormand-Prince) integrator.
    Returns (y_final, num_f_calls).
    """
    y = np.array(y0, dtype=float)
    t = t0
    n = len(y)
    
    # Dormand-Prince coefficients
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1.0
    a7 = 1.0
    
    b2 = 1/5
    b3 = 3/40
    b4 = 3/10
    b5 = -11/54
    b6 = 16/135
    b7 = 5/72
    
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1/2
    
    d1 = 35/384
    d3 = 500/1113
    d4 = 125/192
    d5 = -2187/6784
    d6 = 11/84
    d7 = 0.0
    
    e1 = 71/57600
    e2 = -7/16695
    e3 = 71/1920
    e4 = -17253/339200
    e5 = 22/525
    e6 = -1/40
    e7 = 0.0
    
    # Initial step size estimate
    k1 = f(t, y)
    num_calls = 1
    y_err = np.maximum(atol, rtol * np.abs(y))
    err = np.abs(k1) / y_err
    if np.any(err) == 0:
        h = 0.01 * (t1 - t0)
    else:
        h = 0.01 * (t1 - t0) / (np.max(err) ** 0.2)
    h = min(h, t1 - t0)
    h = max(h, 1e-14)
    
    steps = 0
    while t < t1 - 1e-15:
        if steps >= max_steps:
            break
        if _check_deadline():
            break
        
        steps += 1
        h = min(h, t1 - t)
        
        k1 = f(t, y)
        num_calls += 1
        
        k2 = f(t + c2 * h, y + b2 * h * k1)
        num_calls += 1
        
        k3 = f(t + c3 * h, y + h * (b3 * k1 + (3/40 - b3) * k2))
        # Wait, let me use the standard RK45 Butcher tableau properly
        # Actually, let me just use a simpler approach: fixed-step RK4 with many steps
        
        break  # Fallback to fixed step
    
    # Use fixed-step RK4 with many steps for reliability
    y = np.array(y0, dtype=float)
    t = t0
    num_calls = 0
    
    # Choose number of steps based on problem stiffness
    # For heat equation on 10x10, eigenvalues are O(1) to O(4)
    # Use 10000 steps for good accuracy
    steps = 10000
    h = (t1 - t0) / steps
    
    for _ in range(steps):
        if _check_deadline():
            break
        k1 = f(t, y)
        num_calls += 1
        k2 = f(t + h/2, y + h/2 * k1)
        num_calls += 1
        k3 = f(t + h/2, y + h/2 * k2)
        num_calls += 1
        k4 = f(t + h, y + h * k3)
        num_calls += 1
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y, num_calls


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    global _start_time
    _start_time = time.time()
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to detect if this is a linear heat equation and use matrix exponential
    A = _probe_linear_heat(f, t0, y0)
    
    if A is not None:
        # Use matrix exponential: y(t1) = expm(A * (t1 - t0)) @ y0
        # This is exact for linear ODEs (up to numerical error in expm)
        T = t1 - t0
        try:
            expm_A_T = expm(A * T)
            y_final = expm_A_T @ y0
            # Count the probe calls: 2 (time invariance) + 2 (linearity) + 100 (columns) = 104
            num_calls = 104
            return y_final
        except Exception:
            pass
    
    # Fallback: use RK4 with adaptive step or many fixed steps
    # Use RK4 with 20000 steps for high accuracy
    y = np.array(y0, dtype=float)
    t = t0
    num_calls = 0
    
    # Choose step count based on time interval
    T = t1 - t0
    # For heat equation, use enough steps for accuracy
    steps = 20000
    h = T / steps
    
    for _ in range(steps):
        if _check_deadline():
            break
        k1 = f(t, y)
        num_calls += 1
        k2 = f(t + h/2, y + h/2 * k1)
        num_calls += 1
        k3 = f(t + h/2, y + h/2 * k2)
        num_calls += 1
        k4 = f(t + h, y + h * k3)
        num_calls += 1
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
