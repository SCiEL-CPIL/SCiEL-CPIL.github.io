"""ODE solver for 15-dim stiff linear system using matrix exponential via scaling-and-squaring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _matrix_exp_taylor(A, dt, order=30):
    """Compute exp(A*dt) using Taylor series with scaling-and-squaring."""
    A_scaled = A * dt
    # Scale to reduce norm for better Taylor convergence
    norm_A = np.linalg.norm(A_scaled, ord=np.inf)
    if norm_A < 1e-15:
        return np.eye(A.shape[0])
    
    # Choose s such that ||A_scaled/s|| < 0.5
    s = max(1, int(np.ceil(np.log2(norm_A / 0.5))))
    B = A_scaled / (2 ** s)
    
    # Taylor series: exp(B) = sum_{k=0}^{order} B^k / k!
    n = B.shape[0]
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, order + 1):
        term = term @ B / k
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-16 * np.linalg.norm(result, ord=np.inf):
            break
    
    # Square s times
    for _ in range(s):
        result = result @ result
    
    return result


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    
    # Probe to extract A: f(t, y) should be A @ y for linear constant-coefficient system
    # Use basis vectors to get columns of A
    # First check if there's an offset: f(t, 0)
    f0 = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # Check if it's truly linear (no offset) by verifying f(t, 2*e_i) = 2*f(t, e_i)
    # But to save calls, assume linear: A[:, j] = f(t0, e_j) - f0 if offset exists
    # For pure linear: A[:, j] = f(t0, e_j)
    
    # Probe each basis vector
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        Aj = np.array(f(t0, e_j), dtype=float)
        A[:, j] = Aj - f0  # subtract offset if any
    
    # Verify linearity with a quick check (optional, uses 1 more call)
    # Skip verification to save calls; trust the premise
    
    # Compute matrix exponential exp(A * dt)
    expA_dt = _matrix_exp_taylor(A, dt, order=40)
    
    # Apply to y0
    y_final = expA_dt @ y0
    
    # Deadline guard: if we're running long, we could try a fallback,
    # but with ~16 calls this should be very fast
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: use a simple explicit method with small steps
        # But this shouldn't happen with matrix exponential approach
        pass
    
    return y_final
# EVOLVE-BLOCK-END
