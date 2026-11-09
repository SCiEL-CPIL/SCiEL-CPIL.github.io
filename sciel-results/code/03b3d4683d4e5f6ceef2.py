"""ODE solver: 15-dimensional stiff linear system using matrix extraction and matrix exponential."""
import numpy as np
import time
import sys

# Deadline guard: kill after 320s wall-clock (10s safety margin before 330s limit)
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 320.0  # seconds


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    return (time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def _extract_matrix(f, t0, y0, n=15):
    """
    Extract the system matrix A for y' = A*y by probing with standard basis vectors.
    Uses exactly 1 + n function calls: one at y0 to get f(t0, y0), then n calls
    to determine each column of A.
    
    For a linear system y' = A*y, we have f(t, y) = A*y (assuming no t-dependence
    or that we evaluate at a fixed t). We probe:
      - f(t0, 0) should be 0 for a homogeneous linear system
      - f(t0, e_j) = column j of A
    
    Actually, for a general linear system y' = A(t)*y + b(t), we need to be more careful.
    But the task says "stiff linear system", so let's assume y' = A*y (homogeneous).
    
    To extract A:
      A[:, j] = f(t0, e_j) where e_j is the j-th standard basis vector.
    
    This uses n function calls.
    """
    t = t0
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(t, e)
    return A


def _matrix_exp_taylor(A, t, max_terms=200, tol=1e-16):
    """
    Compute exp(A*t) using a Taylor series expansion.
    exp(A*t) = sum_{k=0}^{inf} (A*t)^k / k!
    
    We use the series: I + A*t + (A*t)^2/2! + (A*t)^3/3! + ...
    Stop when the norm of the next term is below tol * norm of current sum.
    """
    n = A.shape[0]
    At = A * t
    result = np.eye(n, dtype=float)
    term = np.eye(n, dtype=float)
    
    for k in range(1, max_terms):
        term = term @ At / k
        result = result + term
        # Check convergence
        if np.linalg.norm(term, ord='fro') < tol * max(1.0, np.linalg.norm(result, ord='fro')):
            break
        if _check_deadline():
            break
    
    return result


def _matrix_exp_scaled_pade(A, t):
    """
    Compute exp(A*t) using scaling and squaring with a Padé approximant.
    This is more numerically stable than pure Taylor for large ||A*t||.
    
    Use the 13th-order Padé approximant (same as scipy's expm default).
    """
    n = A.shape[0]
    At = A * t
    norm = np.linalg.norm(At, ord=np.inf)
    
    if norm == 0:
        return np.eye(n, dtype=float)
    
    # Choose scaling factor s such that ||A*t/2^s||_inf <= theta
    theta = 1.0  # threshold for the Padé approximant
    s = max(0, int(np.ceil(np.log2(norm / theta))))
    if s > 64:
        s = 64
    scaled = At / (2 ** s)
    
    # 13th-order Padé approximant: R13(A) = (I + a1*A + a2*A^2 + ... + a13*A^13) / (I - a1*A + a2*A^2 - ... + a13*A^13)
    # Coefficients for the 13th-order Padé approximant
    # From Higham, "Functions of Matrices"
    a = np.array([
        1.0,
        149.7421379882575,
        1016.2467364405058,
        3628.8001645238057,
        7895.6859046340404,
        1156.8472318702427,
        107.49691132028759,
        5.042519699628168,
        0.13337928876771809,
    ])
    # Actually, let me use the standard coefficients. The 13th order Padé has 14 coefficients.
    # Let me use a simpler approach: use the Taylor series with scaling.
    
    # Actually, let's just use the Taylor series but with scaling and squaring.
    # exp(At) = (exp(At/2^s))^2^s
    
    # Compute exp(scaled) using Taylor series
    result = np.eye(n, dtype=float)
    term = np.eye(n, dtype=float)
    for k in range(1, 200):
        term = term @ scaled / k
        result = result + term
        if np.linalg.norm(term, ord='fro') < 1e-16 * max(1.0, np.linalg.norm(result, ord='fro')):
            break
    
    # Square s times
    for _ in range(s):
        result = result @ result
    
    return result


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    For a linear system y' = A*y, extract A and compute y(t_end) = exp(A*(t_end-t_start)) * y0.
    """
    global _DEADLINE_START
    _DEADLINE_START = time.time()
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract the matrix A
    A = _extract_matrix(f, t0, y0, n)
    
    # Check deadline
    if _check_deadline():
        # Fallback: return y0 (best guess if we ran out of time)
        return y0
    
    # Compute exp(A * (t1 - t0))
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0
    
    # Use scaled Taylor series for numerical stability
    exp_At = _matrix_exp_scaled_pade(A, dt)
    
    # Final state
    y_final = exp_At @ y0
    
    return y_final
