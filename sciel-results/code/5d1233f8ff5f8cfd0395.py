"""Fourier-parameterized 12-stage product formula for Heisenberg chain evolution."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    dim = 2 ** N
    
    # Identity
    I = np.eye(dim, dtype=complex)
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build bond operators: S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + anisotropy * S_i^z S_{i+1}^z
    # S^+ = (sx + i*sy)/2, S^- = (sx - i*sy)/2, S^z = sz/2
    
    sp = (sx + 1j * sy) / 2
    sm = (sx - 1j * sy) / 2
    szh = sz / 2
    
    def bond_operator(i, j):
        """XXZ bond between sites i and j (0-indexed)."""
        # S_i^+ S_j^- + S_i^- S_j^+ + anisotropy * S_i^z S_j^z
        terms = []
        for k in range(N):
            if k == i:
                terms.append(sp)
            elif k == j:
                terms.append(sm)
            else:
                terms.append(I[0:1, 0:1].reshape(1,1) * np.eye(2, dtype=complex))
        term1 = kron_all(terms)
        
        terms = []
        for k in range(N):
            if k == i:
                terms.append(sm)
            elif k == j:
                terms.append(sp)
            else:
                terms.append(np.eye(2, dtype=complex))
        term2 = kron_all(terms)
        
        terms = []
        for k in range(N):
            if k == i:
                terms.append(szh)
            elif k == j:
                terms.append(szh)
            else:
                terms.append(np.eye(2, dtype=complex))
        term3 = kron_all(terms)
        
        return term1 + term2 + anisotropy * term3
    
    # Odd bonds: (0,1), (2,3), (4,5)
    A_bonds = sum(bond_operator(i, i+1) for i in range(0, N, 2))
    # Even bonds: (1,2), (3,4)
    B_bonds = sum(bond_operator(i, i+1) for i in range(1, N-1, 2))
    
    # Uniform field: h * sum S_i^z, split half to each
    field_op = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        terms = [np.eye(2, dtype=complex)] * N
        terms[site] = szh
        field_op += kron_all(terms)
    
    A = A_bonds + 0.5 * field * field_op
    B = B_bonds + 0.5 * field * field_op
    
    return A, B

def _error_from_coefficients(a, b, A, B, times=(0.25, 0.5, 1.0)):
    """Compute mean spectral norm error for given coefficients."""
    total_err = 0.0
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        diff = U - U_exact
        # Spectral norm = largest singular value
        err = np.linalg.norm(diff, 2)
        total_err += err
    return total_err / len(times)

def coefficients(stages):
    """Fourier-parameterized 12-stage product formula optimized for minimal spectral norm error."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    
    # Fourier parameterization: a_k = 1/12 + sum_{n=1}^{Nf} (alpha_n cos(2*pi*n*k/12) + beta_n sin(2*pi*n*k/12))
    # Same for b_k. This enforces smoothness and sum=1 automatically (DC term fixed to 1/12).
    
    def fourier_to_coeffs(params):
        """Convert Fourier parameters to coefficient arrays."""
        # params layout: [alpha_1..alpha_Nf, beta_1..beta_Nf] for a, then same for b
        Nf = 3  # Number of Fourier harmonics
        k_vals = np.arange(12)
        a = np.full(12, 1.0/12)
        b = np.full(12, 1.0/12)
        
        for n in range(1, Nf+1):
            alpha_a = params[n-1]
            beta_a = params[Nf + n - 1]
            alpha_b = params[2*Nf + n - 1]
            beta_b = params[3*Nf + n - 1]
            
            cos_term = np.cos(2 * np.pi * n * k_vals / 12)
            sin_term = np.sin(2 * np.pi * n * k_vals / 12)
            
            a += alpha_a * cos_term + beta_a * sin_term
            b += alpha_b * cos_term + beta_b * sin_term
        
        # Clamp to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        return a.tolist(), b.tolist()
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a, b = fourier_to_coeffs(params)
        return _error_from_coefficients(a, b, A, B)
    
    # Initial guess: all zeros (uniform coefficients)
    Nf = 3
    x0 = np.zeros(4 * Nf)
    
    best_result = None
    best_score = float('inf')
    
    # Try a few random initializations
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            x0 = np.zeros(4 * Nf)
        else:
            x0 = np.random.normal(0, 0.05, 4 * Nf)
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
        
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Fallback to seed behavior
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    a, b = fourier_to_coeffs(best_result)
    
    # Ensure sums are exactly 1 by normalization
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1) > 1e-12:
        a = [x / sum_a for x in a]
    if abs(sum_b - 1) > 1e-12:
        b = [x / sum_b for x in b]
    
    # Final clamp
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    return a, b
