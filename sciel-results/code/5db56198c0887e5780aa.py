"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

# Build the Heisenberg chain Hamiltonian components
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

# Pre-compute the Hamiltonians and eigen-decompositions for fast expm
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
    if abs(sum(a) - 1.0) > 1e-10 or abs(sum(b) - 1.0) > 1e-10:
        return False
    return True

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using symmetric optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Symmetric structure: a[0..5] determine a[11..6], b[0..5] determine b[11..6]
    # a[11-i] = a[i], b[11-i] = b[i]
    
    best_a_sym = np.full(6, 0.5/6.0)
    best_b_sym = np.full(6, 0.5/6.0)
    
    def expand_sym(sym_a, sym_b):
        a = np.zeros(12)
        b = np.zeros(12)
        for i in range(6):
            a[i] = sym_a[i]
            a[11-i] = sym_a[i]
            b[i] = sym_b[i]
            b[11-i] = sym_b[i]
        return a, b

    def get_error_from_sym(sym_a, sym_b):
        if np.any(np.abs(sym_a) > 50) or np.any(np.abs(sym_b) > 50):
            return 1e10
            
        sum_a = np.sum(sym_a)
        sum_b = np.sum(sym_b)
        
        if abs(sum_a) < 1e-15 or abs(sum_b) < 1e-15:
            return 1e10
            
        norm_a = sym_a * (0.5 / sum_a)
        norm_b = sym_b * (0.5 / sum_b)
        
        if np.any(np.abs(norm_a) > 50) or np.any(np.abs(norm_b) > 50):
            return 1e10
            
        a, b = expand_sym(norm_a, norm_b)
        return _compute_error(a, b)

    def objective(params):
        sym_a = params[:6]
        sym_b = params[6:12]
        return get_error_from_sym(sym_a, sym_b)

    x0 = np.concatenate([best_a_sym, best_b_sym])
    
    n = 12
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.01
        simplex.append(xi)
        
    simplex_vals = [objective(s) for s in simplex]
    order = np.argsort(simplex_vals)
    simplex = [simplex[i] for i in order]
    simplex_vals = [simplex_vals[i] for i in order]
    
    best_val = simplex_vals[0]
    best_x = simplex[0].copy()
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 1000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        f_ref = objective(x_ref)
        
        if f_ref < simplex_vals[0]:
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_vals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                simplex_vals[-1] = f_ref
        elif f_ref < simplex_vals[-2]:
            simplex[-1] = x_ref
            simplex_vals[-1] = f_ref
        else:
            x_con = x_centroid + rho * (x_worst - x_centroid)
            f_con = objective(x_con)
            if f_con < simplex_vals[-1]:
                simplex[-1] = x_con
                simplex_vals[-1] = f_con
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex_vals[i] = objective(simplex[i])
        
        order = np.argsort(simplex_vals)
        simplex = [simplex[i] for i in order]
        simplex_vals = [simplex_vals[i] for i in order]
        
        if simplex_vals[0] < best_val:
            best_val = simplex_vals[0]
            best_x = simplex[0].copy()
            
        if max(simplex_vals) - min(simplex_vals) < 1e-10:
            break
            
    sym_a = best_x[:6]
    sym_b = best_x[6:12]
    
    sum_a = np.sum(sym_a)
    sum_b = np.sum(sym_b)
    
    if abs(sum_a) > 1e-15:
        sym_a = sym_a * (0.5 / sum_a)
    else:
        sym_a = np.full(6, 0.5/6.0)
        
    if abs(sum_b) > 1e-15:
        sym_b = sym_b * (0.5 / sum_b)
    else:
        sym_b = np.full(6, 0.5/6.0)
        
    a, b = expand_sym(sym_a, sym_b)
    
    if not _is_valid(list(a), list(b)):
        return [1.0/12] * 12, [1.0/12] * 12
        
    return list(a), list(b)

# EVOLVE-BLOCK-END
