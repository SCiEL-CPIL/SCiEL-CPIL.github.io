"""Improved program using numerical optimization for product formula coefficients."""
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain with anisotropy 0.8, field 0.3."""
    N = 6
    dim = 2 ** N
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    anisotropy = 0.8
    field = 0.3
    
    # Build full Hamiltonian H = A + B
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # Plus uniform field split half to A, half to B
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds for A
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    for i, j in odd_bonds:
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        A += 0.5 * anisotropy * kron_all(ops)
        ops[i] = sy
        ops[j] = sy
        A += 0.5 * anisotropy * kron_all(ops)
        ops[i] = sz
        ops[j] = sz
        A += kron_all(ops)
    
    # Even bonds for B
    even_bonds = [(1, 2), (3, 4)]
    for i, j in even_bonds:
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        B += 0.5 * anisotropy * kron_all(ops)
        ops[i] = sy
        ops[j] = sy
        B += 0.5 * anisotropy * kron_all(ops)
        ops[i] = sz
        ops[j] = sz
        B += kron_all(ops)
    
    # Uniform field: half to A, half to B
    for site in range(N):
        ops = [I2] * N
        ops[site] = sz
        field_term = 0.5 * field * kron_all(ops)
        A += field_term
        B += field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, times=(0.25, 0.5, 1.0)):
    """Compute mean spectral norm error over given times."""
    dim = A.shape[0]
    errors = []
    for t in times:
        # Compute product formula U(t)
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B)
            B_scaled = -1j * b[k] * t * B
            # Use scipy.linalg.expm if available, else use eigendecomposition
            try:
                from scipy.linalg import expm
                expB = expm(B_scaled)
                expA = expm(-1j * a[k] * t * A)
            except ImportError:
                # Fallback: eigendecomposition
                eigvals_B, eigvecs_B = np.linalg.eig(B_scaled)
                expB = eigvecs_B @ np.diag(np.exp(eigvals_B)) @ np.linalg.inv(eigvecs_B)
                eigvals_A, eigvecs_A = np.linalg.eig(-1j * a[k] * t * A)
                expA = eigvecs_A @ np.diag(np.exp(eigvals_A)) @ np.linalg.inv(eigvecs_A)
            U = expB @ expA @ U
        
        # Compute exact propagator
        try:
            from scipy.linalg import expm
            U_exact = expm(-1j * t * (A + B))
        except ImportError:
            eigvals, eigvecs = np.linalg.eig(-1j * t * (A + B))
            U_exact = eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)
        
        diff = U - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    
    return np.mean(errors)

def coefficients(stages):
    """Optimized product formula coefficients via Nelder-Mead optimization."""
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    
    # Initial guess: symmetric Strang-like splitting
    n = stages
    a = [1.0 / n] * n
    b = [1.0 / n] * n
    
    def objective(params):
        """Objective function for optimization. params has 2*n values: first n for a, next n for b."""
        a_arr = params[:n].copy()
        b_arr = params[n:].copy()
        
        # Enforce sum constraints by projecting
        a_sum = np.sum(a_arr)
        b_sum = np.sum(b_arr)
        if abs(a_sum) > 1e-10:
            a_arr = a_arr / a_sum
        if abs(b_sum) > 1e-10:
            b_arr = b_arr / b_sum
        
        # Clip to valid range
        a_arr = np.clip(a_arr, -50.0, 50.0)
        b_arr = np.clip(b_arr, -50.0, 50.0)
        
        # Re-normalize after clipping
        a_sum = np.sum(a_arr)
        b_sum = np.sum(b_arr)
        if abs(a_sum) > 1e-10:
            a_arr = a_arr / a_sum
        if abs(b_sum) > 1e-10:
            b_arr = b_arr / b_sum
        
        return _spectral_norm_error(a_arr.tolist(), b_arr.tolist(), A, B)
    
    # Try Nelder-Mead optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback: return initial guess
        return a, b
    
    initial_params = a + b
    best_result = None
    best_score = objective(initial_params)
    
    # Run optimization with time check
    max_iter = 200
    
    def callback(xk):
        nonlocal best_score
        score = objective(xk)
        if score < best_score:
            best_score = score
            best_result = xk.copy()
        return (time.time() - start_time) > deadline - 10.0
    
    try:
        result = minimize(objective, initial_params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True, 'callback': callback})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x.copy()
    except Exception:
        pass
    
    # Extract best coefficients
    if best_result is not None:
        a_opt = best_result[:n].copy()
        b_opt = best_result[n:].copy()
    else:
        a_opt = np.array(a)
        b_opt = np.array(b)
    
    # Final normalization and clipping
    a_sum = np.sum(a_opt)
    b_sum = np.sum(b_opt)
    if abs(a_sum) > 1e-10:
        a_opt = a_opt / a_sum
    if abs(b_sum) > 1e-10:
        b_opt = b_opt / b_sum
    
    a_opt = np.clip(a_opt, -50.0, 50.0)
    b_opt = np.clip(b_opt, -50.0, 50.0)
    
    # Re-normalize
    a_sum = np.sum(a_opt)
    b_sum = np.sum(b_opt)
    if abs(a_sum) > 1e-10:
        a_opt = a_opt / a_sum
    if abs(b_sum) > 1e-10:
        b_opt = b_opt / b_sum
    
    return a_opt.tolist(), b_opt.tolist()
