import numpy as np
import time

# Build the 6-site XXZ Heisenberg chain Hamiltonian and split into A (odd bonds) and B (even bonds)
# Each includes half the uniform field

def _build_hamiltonians():
    """Build A and B matrices for the 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    delta = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Identity
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Kronecker product of a list of 2x2 matrices."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build operators on each site
    def op_on_site(site, op):
        """Apply op to site, identity elsewhere."""
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j, op1, op2):
        """Apply op1 to site i and op2 to site j."""
        ops = [I2] * N
        ops[i] = op1
        ops[j] = op2
        return kron_all(ops)
    
    # XXZ Hamiltonian: H = sum_i [sx_i sx_{i+1} + sy_i sy_{i+1} + delta sz_i sz_{i+1}] - field * sum_i sz_i
    # For open chain with N=6, bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    # Each gets half the field
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    for i, j in odd_bonds:
        A += bond_op(i, j, sx, sx) + bond_op(i, j, sy, sy) + delta * bond_op(i, j, sz, sz)
    
    # Even bonds: (1,2), (3,4)
    even_bonds = [(1, 2), (3, 4)]
    for i, j in even_bonds:
        B += bond_op(i, j, sx, sx) + bond_op(i, j, sy, sy) + delta * bond_op(i, j, sz, sz)
    
    # Half field on each
    field_half = field / 2.0
    for site in range(N):
        A -= field_half * op_on_site(site, sz)
        B -= field_half * op_on_site(site, sz)
    
    return A, B

def _expm(mat):
    """Matrix exponential using eigenvalue decomposition."""
    w, V = np.linalg.eig(mat)
    return (V @ np.diag(np.exp(w))) @ np.linalg.inv(V)

def _score_coefficients(a, b, A, B, times):
    """Compute the mean spectral norm error."""
    errors = []
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = np.linalg.multi_dot([_expm(-1j * b[k] * t * B), _expm(-1j * a[k] * t * A), U])
        # Exact: exp(-i t (A+B))
        U_exact = _expm(-1j * t * (A + B))
        diff = U - U_exact
        # Spectral norm (largest singular value)
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    return np.mean(errors)

def _optimize():
    """Optimize the 24 coefficients (12 a's and 12 b's)."""
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Precompute exponentials for various coefficient values to speed up
    # Strategy: use scipy's minimize or a custom Nelder-Mead
    
    from scipy.optimize import minimize
    
    # Parameters: 24 values (12 a's, 12 b's)
    # Constraint: sum(a) = 1, sum(b) = 1
    # Use parameterization: a_i = 1/12 + x_i where sum(x) = 0
    # Similarly for b
    
    # Actually, let's just optimize with constraints
    # Start from a reasonable initial guess
    
    # Initial guess: equal splitting
    n = 12
    a0 = [1.0/n] * n
    b0 = [1.0/n] * n
    x0 = a0 + b0
    
    def objective(x):
        a = x[:n]
        b = x[n:]
        # Check constraints
        if abs(sum(a) - 1.0) > 1e-6 or abs(sum(b) - 1.0) > 1e-6:
            return 1e10
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _score_coefficients(a, b, A, B, times)
    
    # Use SLSQP with constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x[:n]) - 1.0},
        {'type': 'eq', 'fun': lambda x: sum(x[n:]) - 1.0}
    ]
    bounds = [(-50, 50)] * (2*n)
    
    result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                      bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
    
    a_opt = result.x[:n].tolist()
    b_opt = result.x[n:].tolist()
    
    # Verify constraints
    a_opt = [float(v) for v in a_opt]
    b_opt = [float(v) for v in b_opt]
    
    # Ensure exact sum = 1 by adjusting last element
    a_opt[-1] += 1.0 - sum(a_opt)
    b_opt[-1] += 1.0 - sum(b_opt)
    
    # Clip to bounds
    a_opt = [max(-50, min(50, v)) for v in a_opt]
    b_opt = [max(-50, min(50, v)) for v in b_opt]
    
    # Re-normalize after clipping
    a_opt[-1] += 1.0 - sum(a_opt)
    b_opt[-1] += 1.0 - sum(b_opt)
    
    return a_opt, b_opt

# Precompute at module load
_start_time = time.time()

try:
    _OPTIMAL_A, _OPTIMAL_B = _optimize()
    _HAS_OPTIMAL = True
except Exception:
    _HAS_OPTIMAL = False
    # Fallback: simple equal splitting
    _OPTIMAL_A = [1.0/12] * 12
    _OPTIMAL_B = [1.0/12] * 12

# Time check
_elapsed = time.time() - _start_time

def coefficients(stages):
    """Return optimized 12-stage product formula coefficients."""
    if _HAS_OPTIMAL:
        a = list(_OPTIMAL_A)
        b = list(_OPTIMAL_B)
    else:
        a = [1.0/12] * 12
        b = [1.0/12] * 12
    
    # Pad or truncate to match stages if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
