import numpy as np
import time

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

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-10, time_limit=100.0):
    """
    Pure NumPy Nelder-Mead simplex optimizer with bounds and time limit.
    Minimizes f(x) where x is in bounds.
    """
    start_time = time.time()
    n = len(x0)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i, i - 1] += 0.1 * (bounds[i - 1][1] - bounds[i - 1][0]) if bounds[i - 1][1] != bounds[i - 1][0] else 0.1
        simplex[i, i - 1] = np.clip(simplex[i, i - 1], bounds[i - 1][0], bounds[i - 1][1])
    
    # Evaluate initial simplex
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    # Reflection, expansion, contraction coefficients
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:n], axis=0)
        worst = simplex[n]
        worst_f = fvals[n]
        
        # Reflection
        xr = centroid + alpha * (centroid - worst)
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n] = fe
            else:
                simplex[n] = xr
                fvals[n] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[n] = xr
            fvals[n] = fr
        else:
            # Contraction
            if fr < worst_f:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - worst)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc < min(fr, worst_f):
                simplex[n] = xc
                fvals[n] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
        
        # Update best
        best_idx = np.argmin(fvals)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _compute_error_frobenius(a, b):
    """Compute the mean Frobenius norm error over the three time points (faster, smoother)."""
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
        err = np.linalg.norm(U - U_exact, 'fro')
        errors.append(err)
    return np.mean(errors)

def _normalize_params(params):
    """
    Take 24 raw parameters (12 for a, 12 for b), normalize so sum(a)=1, sum(b)=1.
    Returns (a, b) arrays or None if invalid.
    """
    a_raw = params[:12]
    b_raw = params[12:24]
    
    if not np.all(np.isfinite(a_raw)) or not np.all(np.isfinite(b_raw)):
        return None, None
    
    sum_a = np.sum(a_raw)
    sum_b = np.sum(b_raw)
    
    if abs(sum_a) < 1e-12 or abs(sum_b) < 1e-12:
        return None, None
    
    a = a_raw / sum_a
    b = b_raw / sum_b
    
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return None, None
    
    return a, b

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using Nelder-Mead on full 24-param space.
    """
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Fallback: Simple Strang splitting (symmetric, valid)
    fallback_a = np.array([1.0/12.0] * 12)
    fallback_b = np.zeros(12)
    for i in range(0, 12, 2):
        fallback_b[i] = 1.0/6.0
    
    best_params = (list(fallback_a), list(fallback_b))
    best_spectral_error = _compute_error(fallback_a, fallback_b)
    
    # Objective for Nelder-Mead: use Frobenius norm (smooth) on full 24 params
    def objective(params):
        if time.time() - start_time > deadline:
            return 1e10
        a, b = _normalize_params(params)
        if a is None:
            return 1e10
        return _compute_error_frobenius(a, b)
    
    # Initial guess: Strang splitting as raw params (sum will be normalized)
    x0 = np.concatenate([fallback_a, fallback_b])
    
    # Bounds for raw parameters (will be normalized, so bounds are loose)
    bounds = [(-10.0, 10.0)] * 24
    
    # Run Nelder-Mead
    try:
        remaining_time = deadline - time.time() - 5  # leave 5s for final eval
        if remaining_time > 5:
            best_x, best_val = _nelder_mead(objective, x0, bounds, max_iter=1000, tol=1e-12, time_limit=remaining_time)
            
            a_best, b_best = _normalize_params(best_x)
            if a_best is not None:
                # Check spectral norm error for the final answer
                spectral_err = _compute_error(a_best, b_best)
                if spectral_err < best_spectral_error:
                    best_spectral_error = spectral_err
                    best_params = (list(a_best), list(b_best))
    except Exception:
        pass
    
    # Final validation
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
    
    return list(fallback_a), list(fallback_b)

# EVOLVE-BLOCK-END
