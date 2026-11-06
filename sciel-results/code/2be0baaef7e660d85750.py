"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm as sparse_expm
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    anisotropy = 0.8
    field = 0.3
    
    # Build full XXZ Hamiltonian
    # H = sum_j [Jx (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] + h sum_j Sz_j
    # With Jx=Jy=1, Jz=anisotropy, h=field
    # Split into odd bonds (j=0,2,4) and even bonds (j=1,3,5)
    # Each half includes half the field
    
    # Single spin operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    def tensor_op(op, site):
        """Apply operator op to site in n-site system."""
        ops = [np.eye(2)] * n
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def two_site_op(op1, op2, site1, site2):
        """Apply op1 to site1 and op2 to site2."""
        ops = [np.eye(2)] * n
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: j=0,2,4 (between sites 0-1, 2-3, 4-5)
    for j in [0, 2, 4]:
        A += two_site_op(Sx, Sx, j, j+1) + two_site_op(Sy, Sy, j, j+1) + anisotropy * two_site_op(Sz, Sz, j, j+1)
    
    # Even bonds: j=1,3,5 (between sites 1-2, 3-4, 5-0) -- but open chain, so 5-0 doesn't exist
    # For open chain: bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5) -> j=0,2,4
    # Even bonds: (1,2), (3,4) -> j=1,3
    for j in [1, 3]:
        B += two_site_op(Sx, Sx, j, j+1) + two_site_op(Sy, Sy, j, j+1) + anisotropy * two_site_op(Sz, Sz, j, j+1)
    
    # Half field on each
    half_field = field / 2.0
    for j in range(n):
        A += half_field * tensor_op(Sz, j)
        B += half_field * tensor_op(Sz, j)
    
    return A, B

def _compute_error(a, b, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    dim = A.shape[0]
    H = A + B
    errors = []
    
    for t in t_values:
        # Exact propagator
        U_exact = np.linalg.expm(-1j * t * H)
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12
        U_prod = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U_prod = np.linalg.expm(-1j * b[k] * t * B) @ np.linalg.expm(-1j * a[k] * t * A) @ U_prod
        
        err = np.linalg.norm(U_prod - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Optimized 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    t_values = [0.25, 0.5, 1.0]
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    
    # Initial guess: uniform Strang-like
    n = stages
    a = [1.0 / n] * n
    b = [1.0 / n] * n
    
    # Try to optimize using Nelder-Mead or similar
    # Since we have 24 variables with sum constraints, use parameterization
    
    # Check time
    if time.time() > deadline:
        return a, b
    
    # Use a simple local optimization
    # We'll try a few random perturbations and keep the best
    best_a, best_b = a[:], b[:]
    best_err = _compute_error(best_a, best_b, A, B, t_values)
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Try symmetric coefficient patterns
    # Pattern 1: symmetric a and b
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    # with sum = 1
    
    # Try different symmetric patterns
    patterns = []
    
    # Uniform
    patterns.append(([1/12]*12, [1/12]*12))
    
    # Symmetric with emphasis on middle
    for center_weight in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        a_sym = [0.1, 0.15, 0.2, center_weight/6, center_weight/6, 0.2, 0.15, 0.1]
        a_sym = a_sym + a_sym[::-1][:4]  # make it 12 elements
        a_sym = a_sym[:12]
        s = sum(a_sym)
        a_sym = [x/s for x in a_sym]
        b_sym = a_sym[:]
        patterns.append((a_sym, b_sym))
    
    # Try some known high-order splitting coefficients
    # 4th order symmetric: [1/2, 0, 1/2, 0, 1/2, 0, 0, 1/2, 0, 1/2, 0, 1/2] for a
    # This is just repeating Strang
    
    # Let's try a simple grid search over a few parameters
    # a = [alpha, beta, gamma, delta, delta, gamma, gamma, delta, delta, gamma, beta, alpha]
    # b similar
    
    # Actually, let's just do a coarse random search
    np.random.seed(42)
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Random symmetric coefficients
        half_a = np.random.uniform(0.02, 0.2, 6)
        a_candidate = list(half_a) + list(half_a[::-1])
        s = sum(a_candidate)
        a_candidate = [x/s for x in a_candidate]
        
        half_b = np.random.uniform(0.02, 0.2, 6)
        b_candidate = list(half_b) + list(half_b[::-1])
        s = sum(b_candidate)
        b_candidate = [x/s for x in b_candidate]
        
        err = _compute_error(a_candidate, b_candidate, A, B, t_values)
        
        if err < best_err:
            best_err = err
            best_a = a_candidate[:]
            best_b = b_candidate[:]
    
    return best_a, best_b
# EVOLVE-BLOCK-END
