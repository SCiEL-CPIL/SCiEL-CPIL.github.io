"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

# Global deadline guard
_deadline = None

def integrate(f, t_span, y0):
    global _deadline
    
    if _deadline is None:
        _deadline = time.time() + 320.0  # Safety margin before 330s limit
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Check time-invariance and linearity with minimal probes
    # Probe 1: f(t0, y0)
    f_y0 = np.array(f(t0, y0))
    
    # Probe 2: f(t1, y0) to check time-invariance
    f_y0_t1 = np.array(f(t1, y0))
    
    if not np.allclose(f_y0, f_y0_t1, rtol=1e-9, atol=1e-12):
        return _generic_solve(f, t_span, y0)
        
    # Probe 3: f(t0, 0) to check for affine term (should be 0 for linear homogeneous)
    zero_vec = np.zeros(n)
    f_zero = np.array(f(t0, zero_vec))
    
    # If f(0) is not close to 0, it's affine. We can handle y' = A y + b by augmenting or just noting it.
    # For heat equation, b is usually 0. If b != 0, we can solve y' = A y + b.
    # expm(A dt) @ y0 + integral(expm(A (dt-s)) b ds)
    # This is more complex. Let's assume homogeneous first. If f_zero is large, fallback.
    # Actually, we can just reconstruct A and b.
    # f(y) = A y + b
    # f(0) = b
    # f(e_j) = A[:, j] + b
    # So A[:, j] = f(e_j) - b
    
    b = f_zero
    
    # Reconstruct A
    # We need N calls for the columns.
    # Total calls so far: 3 (y0, y0_t1, 0).
    # N calls for columns.
    # Total: N + 3.
    
    # Optimization: If we suspect it's a standard Laplacian, we might not need all N.
    # But N=100 is small. Let's stick to robust reconstruction.
    
    A = np.empty((n, n))
    
    # Check deadline
    if time.time() > _deadline - 5.0:
        return _generic_solve(f, t_span, y0)
        
    for j in range(n):
        if time.time() > _deadline - 5.0:
            # Fallback if running out of time
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j)) - b
        
    # If we broke early, A is incomplete. Fallback.
    if time.time() > _deadline - 5.0:
        return _generic_solve(f, t_span, y0)
        
    # Verify linearity: A @ y0 + b should be close to f_y0
    residual = A @ y0 + b - f_y0
    if not np.allclose(residual, 0, rtol=1e-9, atol=1e-12):
        # It's not linear (or reconstruction failed)
        return _generic_solve(f, t_span, y0)
        
    # Solve y' = A y + b
    # If b is zero, y_end = expm(A dt) @ y0
    # If b is non-zero, y_end = expm(A dt) @ y0 + \int_0^dt expm(A (dt-s)) b ds
    # The integral term is \int_0^dt expm(A tau) b dtau
    # This can be computed as [expm(A dt) - I] @ inv(-A) @ b if A is invertible?
    # Or using block matrix exponential:
    # [ A  b ]
    # [ 0  0 ]
    # expm(dt * Block) @ [y0; 1] -> [y_end; 1]
    
    if np.allclose(b, 0, rtol=1e-9, atol=1e-12):
        # Homogeneous
        try:
            # Use sparse if possible for speed/accuracy?
            # A is dense 100x100. expm_multiply is fine.
            y_end = expm_multiply(dt * A, y0)
        except Exception:
            try:
                E = expm(dt * A)
                y_end = E @ y0
            except Exception:
                return _generic_solve(f, t_span, y0)
    else:
        # Affine
        # Construct block matrix
        # M = [[A, b], [0, 0]]
        # dim = n+1
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is 0
        
        y0_aug = np.zeros(n+1)
        y0_aug[:n] = y0
        y0_aug[n] = 1.0
        
        try:
            # expm_multiply on dense matrix
            # scipy.sparse.linalg.expm_multiply expects sparse or dense
            # It works with dense numpy arrays too, treating them as dense.
            y_end_aug = expm_multiply(dt * M, y0_aug)
            y_end = y_end_aug[:n]
        except Exception:
            try:
                E = expm(dt * M)
                y_end_aug = E @ y0_aug
                y_end = y_end_aug[:n]
            except Exception:
                return _generic_solve(f, t_span, y0)
                
    if not np.all(np.isfinite(y_end)):
        return _generic_solve(f, t_span, y0)
        
    return y_end


def _generic_solve(f, t_span, y0):
    """Fallback generic solver if structure detection fails."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Use RK4 with adaptive or fixed small step.
    # For 100 steps, error is small.
    # We can increase steps if we have time.
    steps = 100
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        if time.time() > _deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
