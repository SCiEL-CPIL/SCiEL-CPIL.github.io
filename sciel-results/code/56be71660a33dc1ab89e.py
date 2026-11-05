"""Optimized 12-stage product formula for 6-site XXZ Heisenberg chain.
Uses scipy differential_evolution to find coefficients minimizing mean spectral-norm error."""

import time
import numpy as np
from scipy.optimize import differential_evolution

_START = time.time()
_DEADLINE = _START + 120.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Each bond: gamma/2*(Sx*Sx + Sy*Sy) + 1/2*Sz*Sz + h/2*(Sz_i + Sz_j)
    Split: A = sum over odd bonds (1-2, 3-4, 5-6), B = sum over even bonds (2-3, 4-5)
    Each includes half the uniform field."""
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site in N-site chain."""
        result = np.array([[1]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    def bond_op(i, j):
        """Two-spin bond operator between sites i and j (0-indexed).
        gamma/2*(Sx_i*Sx_j + Sy_i*Sy_j) + 1/2*Sz_i*Sz_j + h/2*(Sz_i + Sz_j)"""
        Sxi = op_on_site(Sx, i)
        Sxj = op_on_site(Sx, j)
        Syi = op_on_site(Sy, i)
        Syj = op_on_site(Sy, j)
        Szi = op_on_site(Sz, i)
        Szj = op_on_site(Sz, j)
        
        return (gamma/2)*(Sxi @ Sxj + Syi @ Syj) + 0.5*(Szi @ Szj) + h/2*(Szi + Szj)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    B = bond_op(1, 2) + bond_op(3, 4)
    
    return A, B

def _compute_error(coeffs, A, B):
    """Compute mean spectral-norm error for given coefficients at t=0.25, 0.5, 1.0."""
    a, b = coeffs
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    total_err = 0.0
    for t in [0.25, 0.5, 1.0]:
        U = I.copy()
        for k in range(12):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                # Use scipy's expm for accuracy
                from scipy.linalg import expm
                U = expm(-1j * b[k] * t * B) @ U
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                from scipy.linalg import expm
                U = expm(-1j * a[k] * t * A) @ U
        
        # Exact propagator
        from scipy.linalg import expm
        U_exact = expm(-1j * t * (A + B))
        
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    
    return total_err / 3.0

def coefficients(stages):
    """Return optimized (a, b) lists of length `stages` (12)."""
    global _START, _DEADLINE
    
    if stages != 12:
        # Fallback: simple uniform splitting
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Parameterize: 24 free parameters, then enforce sum constraints
    # a has 12 values summing to 1, b has 12 values summing to 1
    # Free params: a[0..10] (11 values), b[0..10] (11 values) -> 22 free
    # a[11] = 1 - sum(a[0..10]), b[11] = 1 - sum(b[0..10])
    
    def objective(x):
        if time.time() > _DEADLINE - 10:
            # Return a large value to signal timeout
            return 1e10
        a = list(x[:11]) + [1.0 - sum(x[:11])]
        b = list(x[11:22]) + [1.0 - sum(x[11:22])]
        
        # Check bounds
        for val in a + b:
            if val < -50 or val > 50:
                return 1e10
        
        return _compute_error((a, b), A, B)
    
    # Initial guess: uniform
    x0 = [1.0/12] * 22
    
    # Bounds for each free parameter
    bounds = [(-50, 50)] * 22
    
    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        
        a = list(result.x[:11]) + [1.0 - sum(result.x[:11])]
        b = list(result.x[11:22]) + [1.0 - sum(result.x[11:22])]
        
        # Clamp to [-50, 50]
        a = [max(-50, min(50, v)) for v in a]
        b = [max(-50, min(50, v)) for v in b]
        
        # Renormalize after clamping
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-12:
            a = [v / sa for v in a]
        if abs(sb) > 1e-12:
            b = [v / sb for v in b]
        
        # Final check
        if abs(sum(a) - 1) > 1e-8 or abs(sum(b) - 1) > 1e-8:
            a = [1.0/12] * 12
            b = [1.0/12] * 12
        
        return a, b
        
    except Exception:
        # Fallback to simple splitting
        a = [1.0/12] * 12
        b = [1.0/12] * 12
        return a, b
