import numpy as np
import time
import math
# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8  # anisotropy
    h = 0.3      # field
    
    # Pauli matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def tensor_product(matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result
    
    # Build operators for each site
    def site_op(op, site):
        ops = [ident] * N
        ops[site] = op
        return tensor_product(ops)
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        A += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        A += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        A += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        A -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    A = (A + A.conj().T) / 2.0

    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        B += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        B += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        B += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        B -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    B = (B + B.conj().T) / 2.0

    return A, B

# Pre-compute the Hamiltonians and eigen-decompositions
_A, _B = _build_hamiltonians()
_H = _A + _B

# Eigen-decompositions for Hermitian matrices: H = V diag(w) V^dagger
# exp(-i t H) = V diag(exp(-i t w)) V^dagger
_wA, _VA = np.linalg.eigh(_A)
_wB, _VB = np.linalg.eigh(_B)
_wH, _VH = np.linalg.eigh(_H)

_T = [0.25, 0.5, 1.0]

def _expm_hermitian_scaled(w, V, scale):
    """Compute exp(-i * scale * H) given eigenvalues w and eigenvectors V."""
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

# Pre-compute exact propagators
_exact_U = [ _expm_hermitian_scaled(_wH, _VH, t) for t in _T ]

def _compute_error(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    dim = 64
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U_B = _expm_hermitian_scaled(_wB, _VB, b[k] * t)
                U = U_B @ U
            if a[k] != 0.0:
                U_A = _expm_hermitian_scaled(_wA, _VA, a[k] * t)
                U = U_A @ U
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _is_valid(a, b):
    """Check if coefficients are valid."""
    if len(a) != 12 or len(b) != 12:
        return False
    for x in a + b:
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return False
    return True

def _nelder_mead(f, x0, bounds, max_iter=200, time_limit=120.0):
    """
    Nelder-Mead optimization with time limit.
    """
    start_time = time.time()
    n = len(x0)
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.1
        if bounds[i][1] > bounds[i][0]:
            step = (bounds[i][1] - bounds[i][0]) * 0.1
        x_new[i] += step
        simplex.append(x_new)
    
    # Clip initial simplex to bounds
    for i in range(n + 1):
        for j in range(n):
            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
    
    # Evaluate initial function values
    f_values = [f(simplex[i]) for i in range(n + 1)]
    
    best_x = simplex[f_values.index(min(f_values))].copy()
    best_f = min(f_values)
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[n])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[n] = x_e
                f_values[n] = f_e
            else:
                simplex[n] = x_r
                f_values[n] = f_r
        elif f_r < f_values[n-1]:
            simplex[n] = x_r
            f_values[n] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[n] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            if f_c < f_values[n]:
                simplex[n] = x_c
                f_values[n] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    f_values[i] = f(simplex[i])
        
        # Update best
        current_best_idx = f_values.index(min(f_values))
        if f_values[current_best_idx] < best_f:
            best_f = f_values[current_best_idx]
            best_x = simplex[current_best_idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Fallback: Simple Strang splitting
    fallback_a = [1.0/12.0] * 12
    fallback_b = [0.0] * 12
    for i in range(0, 12, 2):
        fallback_b[i] = 1.0/6.0
    
    best_params = (fallback_a, fallback_b)
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Bounds for each parameter
    bounds = [(-50, 50)] * 24
    
    # Objective function that takes a 24-dim vector and returns error
    def objective(x):
        a = x[:12]
        b = x[12:]
        
        # Normalize to ensure sum(a)=1, sum(b)=1
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        if abs(sum_a) < 1e-12 or abs(sum_b) < 1e-12:
            return 1e6  # Large penalty
        
        a_norm = a / sum_a
        b_norm = b / sum_b
        
        # Check bounds
        if np.any(a_norm < -50) or np.any(a_norm > 50) or np.any(b_norm < -50) or np.any(b_norm > 50):
            return 1e6
        
        return _compute_error(a_norm, b_norm)
    
    # Initial guess: flatten the fallback
    x0 = np.concatenate([np.array(fallback_a), np.array(fallback_b)])
    
    # Run Nelder-Mead
    try:
        x_best, f_best = _nelder_mead(objective, x0, bounds, max_iter=300, time_limit=120.0)
        
        a_best = x_best[:12]
        b_best = x_best[12:]
        
        # Normalize
        sum_a = np.sum(a_best)
        sum_b = np.sum(b_best)
        
        if abs(sum_a) > 1e-12 and abs(sum_b) > 1e-12:
            a_best = a_best / sum_a
            b_best = b_best / sum_b
            
            if _is_valid(list(a_best), list(b_best)):
                err = _compute_error(a_best, b_best)
                if err < best_error:
                    best_error = err
                    best_params = (list(a_best), list(b_best))
    except Exception:
        pass
    
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
        
    return fallback_a, fallback_b

# EVOLVE-BLOCK-END
