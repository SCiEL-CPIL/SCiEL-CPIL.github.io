"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time

_t_start = time.time()
_DEADLINE = 320.0  # seconds, 10s safety margin before 330s kill


def _check_time():
    return (time.time() - _t_start) < _DEADLINE


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Estimate the linear operator A by probing with basis vectors
    # f(t, y) = A @ y
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        if not _check_time():
            # Fallback: use expm directly if we're running out of time
            dt = t1 - t0
            M = sla.expm(A * dt)
            return M @ y0
        
    dt = t1 - t0
    At = A * dt
    
    # Try eigenvalue decomposition for potentially better accuracy
    # Check if A is symmetric (common for heat equation)
    if np.allclose(At, At.T, atol=1e-12):
        try:
            eigvals, eigvecs = np.linalg.eigh(At)
            exp_eigvals = np.exp(eigvals)
            M = (eigvecs * exp_eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            M = sla.expm(At)
    else:
        try:
            eigvals, eigvecs = np.linalg.eig(At)
            exp_eigvals = np.exp(eigvals)
            M = eigvecs @ np.diag(exp_eigvals) @ np.linalg.inv(eigvecs)
        except np.linalg.LinAlgError:
            M = sla.expm(At)
    
    if not _check_time():
        return M @ y0
    
    y_end = M @ y0
    return y_end
# EVOLVE-BLOCK-END
