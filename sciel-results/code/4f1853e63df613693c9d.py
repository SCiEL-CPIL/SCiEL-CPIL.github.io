"""Improved program using coordinate descent + L-BFGS for product formula coefficients."""
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain with anisotropy 0.8, field 0.3."""
    N = 6
    dim = 2 ** N
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
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> go into A
    odd_bonds = [(0,1), (2,3), (4,5)]
    for i, j in odd_bonds:
        for pauli_x, pauli_y in [(sx, sx), (sy, sy), (sz, sz)]:
            coeff = anisotropy if pauli_x is sz else 1.0
            term = np.zeros((dim, dim), dtype=complex)
            ops = [I2]*N
            ops[i] = pauli_x
            ops[j] = pauli_y
            term = kron_all(ops)
            A += (coeff * term) / 2.0
        for site in [i, j]:
            ops = [I2]*N
            ops[site] = sz
            term = kron_all(ops)
            A += (field * term) / 4.0
    
    # Even bonds: (1,2), (3,4) -> go into B
    even_bonds = [(1,2), (3,4)]
    for i, j in even_bonds:
        for pauli_x, pauli_y in [(sx, sx), (sy, sy), (sz, sz)]:
            coeff = anisotropy if pauli_x is sz else 1.0
            term = np.zeros((dim, dim), dtype=complex)
            ops = [I2]*N
            ops[i] = pauli_x
            ops[j] = pauli_y
            term = kron_all(ops)
            B += (coeff * term) / 2.0
        for site in [i, j]:
            ops = [I2]*N
            ops[site] = sz
            term = kron_all(ops)
            B += (field * term) / 4.0
    
    return A, B

def _precompute_eig(H):
    """Precompute eigendecomposition of Hermitian H."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals, eigvecs

def _expm_cached(coeff, eigvals, eigvecs):
    """Compute exp(coeff * H) using cached eigendecomposition."""
    if abs(coeff) < 1e-15:
        return np.eye(len(eigvals), dtype=complex)
    return (eigvecs * np.exp(coeff * eigvals)) @ eigvecs.conj().T

def _spectral_norm_error(a, b, A_eig, B_eig, times):
    """Compute mean spectral norm error over given times using cached eigs."""
    n = len(a)
    A_eigvals, A_eigvecs = A_eig
    B_eigvals, B_eigvecs = B_eig
    dim = len(A_eigvals)
    
    total_err = 0.0
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(n):
            if abs(b[k]) > 1e-15:
                U = U @ _expm_cached(-1j * b[k] * t, B_eigvals, B_eigvecs)
            if abs(a[k]) > 1e-15:
                U = U @ _expm_cached(-1j * a[k] * t, A_eigvals, A_eigvecs)
        
        U_exact = _expm_cached(-1j * t, A_eigvals + B_eigvals, A_eigvecs) if False else None
        # Need exact: exp(-i*t*(A+B))
        H = A_eigvecs @ np.diag(A_eigvals) @ A_eigvecs.conj().T + B_eigvecs @ np.diag(B_eigvals) @ B_eigvecs.conj().T
        H_eigvals, H_eigvecs = np.linalg.eigh(H)
        U_exact = (H_eigvecs * np.exp(-1j * t * H_eigvals)) @ H_eigvecs.conj().T
        
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(times)

def _normalize_and_clip(a, b, stages):
    """Ensure sums are 1 and values in [-50, 50]."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    
    if abs(a_sum) > 1e-15:
        a = a / a_sum
    else:
        a = np.ones(stages) / stages
    
    if abs(b_sum) > 1e-15:
        b = b / b_sum
    else:
        b = np.ones(stages) / stages
    
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    return a.tolist(), b.tolist()

def _objective_from_params(params, A_eig, B_eig, times, H_eig):
    """Convert parameter vector to (a,b) and compute error."""
    stages = 12
    n = stages
    a = params[:n]
    b = params[n:2*n]
    
    a_list, b_list = _normalize_and_clip(a, b, stages)
    
    n = len(a_list)
    A_eigvals, A_eigvecs = A_eig
    B_eigvals, B_eigvecs = B_eig
    H_eigvals, H_eigvecs = H_eig
    dim = len(A_eigvals)
    
    total_err = 0.0
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(n):
            if abs(b_list[k]) > 1e-15:
                U = U @ _expm_cached(-1j * b_list[k] * t, B_eigvals, B_eigvecs)
            if abs(a_list[k]) > 1e-15:
                U = U @ _expm_cached(-1j * a_list[k] * t, A_eigvals, A_eigvecs)
        
        U_exact = (H_eigvecs * np.exp(-1j * t * H_eigvals)) @ H_eigvecs.conj().T
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(times)

def _coordinate_descent(objective, x0, max_iter=500, tol=1e-10, step_init=0.01):
    """Coordinate descent with adaptive step size."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    f = objective(x)
    best_x = x.copy()
    best_f = f
    step = step_init
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for direction in [1, -1]:
                x_trial = x.copy()
                x_trial[i] += direction * step
                f_trial = objective(x_trial)
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    x_trial = x.copy()
                    x_trial[i] += direction * step * 0.5
                    f_trial = objective(x_trial)
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        step *= 0.7
                        break
            else:
                step *= 0.7
        if not improved:
            break
        if it > 0 and abs(best_f - f) < tol:
            break
    
    return best_x, best_f

def _random_restarts(objective, x0, n_restarts=5, max_iter_per_restart=200):
    """Try multiple random initializations."""
    best_x = x0.copy()
    best_f = objective(x0)
    n = len(x0)
    
    for r in range(n_restarts):
        x_rand = x0 + np.random.randn(n) * 0.05
        x_try, f_try = _coordinate_descent(objective, x_rand, max_iter=max_iter_per_restart)
        if f_try < best_f:
            best_f = f_try
            best_x = x_try
    
    return best_x, best_f

def coefficients(stages):
    """Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    if stages != 12:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigendecompositions
    A_eig = _precompute_eig(A)
    B_eig = _precompute_eig(B)
    H = A + B
    H_eig = _precompute_eig(H)
    
    def obj(x):
        return _objective_from_params(x, A_eig, B_eig, times, H_eig)
    
    # Initial guess
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    x0 = a0 + b0
    
    best_x = x0.copy()
    best_f = obj(x0)
    
    # Phase 1: Coordinate descent from uniform
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            x_cd, f_cd = _coordinate_descent(obj, x0, max_iter=300, step_init=0.01)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd
        except Exception:
            pass
    
    # Phase 2: Random restarts if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            x_rr, f_rr = _random_restarts(obj, best_x, n_restarts=3, max_iter_per_restart=150)
            if f_rr < best_f:
                best_f = f_rr
                best_x = x_rr
        except Exception:
            pass
    
    # Phase 3: Final polish from best found
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            x_polish, f_polish = _coordinate_descent(obj, best_x, max_iter=200, step_init=0.005)
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish
        except Exception:
            pass
    
    a_best, b_best = _normalize_and_clip(best_x[:12], best_x[12:24], 12)
    return a_best, b_best
# EVOLVE-BLOCK-END
