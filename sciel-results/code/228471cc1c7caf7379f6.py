<<<<<<< SEARCH
import numpy as np
import time

def _matrix_exp_hermitian(H, t):
    """Compute exp(-1j * t * H) for Hermitian H."""
    vals, vecs = np.linalg.eigh(H)
    # exp(-1j * t * vals)
    exp_vals = np.exp(-1j * t * vals)
    return vecs @ np.diag(exp_vals) @ vecs.conj().T

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2**n
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    Jz = 1.0
    Jxy = delta * Jz
    
    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)
    
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)
    
    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)
    
    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, t):
    """Compute spectral norm of U(t) - exp(-i t (A+B))."""
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        # Order: exp(-i b_k t B) @ exp(-i a_k t A)
        # Note: The task says U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        # This usually means U = (M_12 ... M_1) where M_k = exp(-i b_k t B) exp(-i a_k t A)
        # Or is it U = M_1 M_2 ... M_12?
        # "in order k = 1..12" implies the product is formed by multiplying from left to right or right to left?
        # Standard convention for "product formula" U = prod_{k=1}^N U_k often means U = U_N ... U_1 if acting on a state vector from the right, or U_1 U_2 ... U_N if the first factor is applied first.
        # Let's check the seed code:
        # U = expm(...) @ expm(...) @ U
        # This accumulates on the left. So U_final = M_12 ... M_1.
        # This means M_1 is applied first to the state.
        U = _matrix_exp_hermitian(B, -b[k] * t) @ _matrix_exp_hermitian(A, -a[k] * t) @ U
        
    U_exact = _matrix_exp_hermitian(A + B, -t)
    diff = U - U_exact
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]
=======
import numpy as np
import time

# Cache for eigen-decompositions
_eig_cache = {}

def _get_eig(H, name):
    """Get cached eigen-decomposition of H."""
    if name not in _eig_cache:
        vals, vecs = np.linalg.eigh(H)
        _eig_cache[name] = (vals, vecs)
    return _eig_cache[name]

def _matrix_exp_hermitian_cached(H, t, name):
    """Compute exp(-1j * t * H) using cached eigen-decomposition."""
    vals, vecs = _get_eig(H, name)
    exp_vals = np.exp(-1j * t * vals)
    return vecs @ np.diag(exp_vals) @ vecs.conj().T

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2**n
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    Jz = 1.0
    Jxy = delta * Jz
    
    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)
        
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)
    
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)
    
    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)
    
    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, A_plus_B, t):
    """Compute spectral norm of U(t) - exp(-i t (A+B)) using cached eigen-decompositions."""
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        U = _matrix_exp_hermitian_cached(B, -b[k] * t, 'B') @ _matrix_exp_hermitian_cached(A, -a[k] * t, 'A') @ U
        
    U_exact = _matrix_exp_hermitian_cached(A_plus_B, -t, 'AB')
    diff = U - U_exact
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]
>>>>>>> REPLACE

