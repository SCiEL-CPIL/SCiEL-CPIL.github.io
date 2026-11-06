"""Optimized 12-stage product formula for 6-site Heisenberg chain time evolution."""
import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonians():
    """Build the even-bond (A) and odd-bond (B) Hamiltonians for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    splus = sx + 1j*sy
    sminus = sx - 1j*sy
    id2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    dim = 2**N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Uniform field: split half to A, half to B
    for i in range(N):
        ops = [id2]*N
        ops[i] = sz
        field_op = kron_all(ops)
        A += 0.5 * field * field_op
        B += 0.5 * field * field_op
    
    # XXZ interaction: J_x (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + J_z Sz_i Sz_{i+1}
    # J_x = 1, J_z = anisotropy = 0.8
    Jx = 1.0
    Jz = anisotropy
    
    for i in range(N-1):
        # Sx_i Sx_{i+1}
        ops = [id2]*N
        ops[i] = sx
        ops[i+1] = sx
        A += Jx * kron_all(ops) if i % 2 == 0 else 0
        B += Jx * kron_all(ops) if i % 2 == 1 else 0
        
        # Sy_i Sy_{i+1}
        ops = [id2]*N
        ops[i] = sy
        ops[i+1] = sy
        A += Jx * kron_all(ops) if i % 2 == 0 else 0
        B += Jx * kron_all(ops) if i % 2 == 1 else 0
        
        # Sz_i Sz_{i+1}
        ops = [id2]*N
        ops[i] = sz
        ops[i+1] = sz
        A += Jz * kron_all(ops) if i % 2 == 0 else 0
        B += Jz * kron_all(ops) if i % 2 == 1 else 0
    
    return A, B

def _spectral_norm_error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    n = len(a)
    errors = []
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(n):
            # exp(-i * b_k * t * B)
            eigvals_b, eigvecs_b = np.linalg.eigh(B)
            expB = eigvecs_b @ np.diag(np.exp(-1j * b[k] * t * eigvals_b)) @ eigvecs_b.conj().T
            U = expB @ U
            
            # exp(-i * a_k * t * A)
            eigvals_a, eigvecs_a = np.linalg.eigh(A)
            expA = eigvecs_a @ np.diag(np.exp(-1j * a[k] * t * eigvals_a)) @ eigvecs_a.conj().T
            U = expA @ U
        
        # Exact: exp(-i * t * (A+B))
        eigvals_ab, eigvecs_ab = np.linalg.eigh(A + B)
        U_exact = eigvecs_ab @ np.diag(np.exp(-1j * t * eigvals_ab)) @ eigvecs_ab.conj().T
        
        diff = U - U_exact
        # Spectral norm = largest singular value
        sv = np.linalg.svd(diff, compute_uv=False)
        errors.append(sv[0])
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigendecompositions for speed
    eigvals_a, eigvecs_a = np.linalg.eigh(A)
    eigvals_b, eigvecs_b = np.linalg.eigh(B)
    eigvals_ab, eigvecs_ab = np.linalg.eigh(A + B)
    
    def _error_fast(a, b):
        """Fast error computation using precomputed eigendecomp."""
        n = len(a)
        errors = []
        for t in times:
            U = np.eye(64, dtype=complex)
            for k in range(n):
                # exp(-i * b_k * t * B)
                expB = eigvecs_b @ np.diag(np.exp(-1j * b[k] * t * eigvals_b)) @ eigvecs_b.conj().T
                U = expB @ U
                # exp(-i * a_k * t * A)
                expA = eigvecs_a @ np.diag(np.exp(-1j * a[k] * t * eigvals_a)) @ eigvecs_a.conj().T
                U = expA @ U
            
            U_exact = eigvecs_ab @ np.diag(np.exp(-1j * t * eigvals_ab)) @ eigvecs_ab.conj().T
            diff = U - U_exact
            sv = np.linalg.svd(diff, compute_uv=False)
            errors.append(sv[0])
        return np.mean(errors)
    
    # Symmetric ansatz: a_k = a_{13-k}, b_k = b_{13-k}
    # This gives 6 free a's and 6 free b's = 12 parameters
    # With constraints sum(a)=1 and sum(b)=1, we have 10 free params
    
    def _symmetric_coeffs(params):
        """Reconstruct full a, b from symmetric parameters."""
        # params[0:6] = a[0:6], a[6:12] = a[0:6][::-1]
        # params[6:12] = b[0:6], b[6:12] = b[0:6][::-1]
        a_half = params[0:6]
        b_half = params[6:12]
        a = list(a_half) + list(a_half[::-1])
        b = list(b_half) + list(b_half[::-1])
        return np.array(a), np.array(b)
    
    def _objective(params):
        a, b = _symmetric_coeffs(params)
        # Enforce sum constraints via penalty
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 1e6 * (sum_a - 1.0)**2 + 1e6 * (sum_b - 1.0)**2
        # Clip to valid range
        a_clipped = np.clip(a, -50, 50)
        b_clipped = np.clip(b, -50, 50)
        err = _error_fast(a_clipped, b_clipped)
        return err + penalty
    
    # Initial guess: uniform splitting with symmetry
    # a_k = 1/12 for all k, b_k = 1/12 for all k
    a_init = np.ones(6) / 12.0
    b_init = np.ones(6) / 12.0
    x0 = np.concatenate([a_init, b_init])
    
    best_params = x0.copy()
    best_err = _objective(x0)
    
    # Try L-BFGS-B
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        # Not enough time for more optimization
        a, b = _symmetric_coeffs(best_params)
        a = np.clip(a, -50, 50).tolist()
        b = np.clip(b, -50, 50).tolist()
        # Normalize
        sa = sum(a)
        sb = sum(b)
        if sa != 0:
            a = [x/sa for x in a]
        if sb != 0:
            b = [x/sb for x in b]
        return a, b
    
    # Try a second optimization with different initial guess
    # Try 4th-order-like coefficients
    # For 4-stage 4th order: a = [1/4, 1/4, 1/4, 1/4], b = [0, 1/2, 0, 1/2] type
    # For symmetric 12-stage, try various patterns
    
    # Try Nelder-Mead as fallback
    try:
        result2 = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if result2.fun < best_err:
            best_err = result2.fun
            best_params = result2.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        a, b = _symmetric_coeffs(best_params)
        a = np.clip(a, -50, 50).tolist()
        b = np.clip(b, -50, 50).tolist()
        sa = sum(a)
        sb = sum(b)
        if sa != 0:
            a = [x/sa for x in a]
        if sb != 0:
            b = [x/sb for x in b]
        return a, b
    
    # Final polish with tighter tolerance
    try:
        result3 = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result3.fun < best_err:
            best_err = result3.fun
            best_params = result3.x.copy()
    except Exception:
        pass
    
    a, b = _symmetric_coeffs(best_params)
    a = np.clip(a, -50, 50).tolist()
    b = np.clip(b, -50, 50).tolist()
    
    # Final normalization to ensure sum = 1 exactly
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
    
    return a, b
