"""Optimized 12-stage product formula for 6-site Heisenberg chain time evolution."""
import numpy as np
from scipy import linalg

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Single-site operators
    # Sx = 0.5 * [[0,1],[1,0]], Sy = 0.5 * [[0,-1j],[1j,0]], Sz = 0.5 * [[1,0],[0,-1]]
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Kronecker product of a list of operators."""
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build the full Hamiltonian H = A + B
    # XXZ chain: H = sum_j [Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + anisotropy * Sz_j Sz_{j+1}] + field * sum_j Sz_j
    # Split into odd bonds (j=0,2,4) -> A and even bonds (j=1,3,5... wait, 6 sites means bonds 0-1,1-2,2-3,3-4,4-5)
    # Odd bonds: (0,1), (2,3), (4,5) -> these are bonds at even indices if we number bonds 0..4
    # Even bonds: (1,2), (3,4) -> wait, let me reconsider
    # 6 sites: bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # "Odd bonds" typically means bonds connecting site 0-1, 2-3, 4-5 (even-indexed sites to next)
    # "Even bonds" means bonds connecting site 1-2, 3-4 (odd-indexed sites to next)
    # Actually, the standard split for a chain is:
    # A: bonds (0,1), (2,3), (4,5) - "odd" in some conventions
    # B: bonds (1,2), (3,4) - "even" in some conventions
    # Plus half the field in each
    
    dim = 2 ** N
    
    # Build A: odd bonds + half field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field term: field * sum_j Sz_j, split half to A and half to B
    for j in range(N):
        ops = [I2] * N
        ops[j] = Sz
        sz_j = kron_all(ops)
        A += (field / 2.0) * sz_j
        B += (field / 2.0) * sz_j
    
    # Bond terms
    for j in range(N - 1):
        # Sx_j Sx_{j+1}
        ops_x = [I2] * N
        ops_x[j] = Sx
        ops_x[j+1] = Sx
        sxj_sxjp1 = kron_all(ops_x)
        
        # Sy_j Sy_{j+1}
        ops_y = [I2] * N
        ops_y[j] = Sy
        ops_y[j+1] = Sy
        syj_sypj1 = kron_all(ops_y)
        
        # Sz_j Sz_{j+1}
        ops_z = [I2] * N
        ops_z[j] = Sz
        ops_z[j+1] = Sz
        szj_szjp1 = kron_all(ops_z)
        
        bond_term = sxj_sxjp1 + syj_sypj1 + anisotropy * szj_szjp1
        
        # Split: even bond index j (0,2,4) goes to A, odd bond index j (1,3) goes to B
        # j=0: bond (0,1) -> A
        # j=1: bond (1,2) -> B
        # j=2: bond (2,3) -> A
        # j=3: bond (3,4) -> B
        # j=4: bond (4,5) -> A
        if j % 2 == 0:
            A += bond_term
        else:
            B += bond_term
    
    return A, B

def _objective(params, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    stages = 12
    a = params[:stages]
    b = params[stages:]
    
    # Check constraints
    if abs(sum(a) - 1.0) > 1e-10 or abs(sum(b) - 1.0) > 1e-10:
        return 1e10
    
    # Check bounds
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    errors = []
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(2**6, dtype=complex)
        for k in range(stages):
            if abs(b[k]) > 1e-15:
                U = linalg.expm(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = linalg.expm(-1j * a[k] * t * A) @ U
        
        # Exact propagator
        H = A + B
        U_exact = linalg.expm(-1j * t * H)
        
        # Spectral norm of difference
        diff = U - U_exact
        err = linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def _optimize():
    """Run optimization to find best coefficients."""
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    stages = 12
    
    # Parameterization: use 22 free parameters + 2 determined by constraints
    # a = a_free + (1 - sum(a_free)) / stages * ones  -> this doesn't work directly
    # Better: a[12] = 1 - sum(a[0:11]), b[12] = 1 - sum(b[0:11])
    
    def constrained_objective(x):
        a = x[:11].copy()
        b = x[11:22].copy()
        a = np.append(a, 1.0 - np.sum(a))
        b = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        
        return _objective(np.concatenate([a, b]), A, B, times)
    
    # Initial guess: Strang-like
    x0 = np.zeros(22)
    for k in range(11):
        x0[k] = 1.0 / 12
        x0[k + 11] = 1.0 / 12
    
    # Use SLSQP with bounds
    bounds = [(-50, 50)] * 22
    
    from scipy.optimize import minimize
    result = minimize(constrained_objective, x0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-12})
    
    a = result.x[:11].copy()
    b = result.x[11:22].copy()
    a = np.append(a, 1.0 - np.sum(a))
    b = np.append(b, 1.0 - np.sum(b))
    
    # Clip to valid range
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize
    a = a / np.sum(a) if abs(np.sum(a)) > 1e-15 else np.ones(12) / 12
    b = b / np.sum(b) if abs(np.sum(b)) > 1e-15 else np.ones(12) / 12
    
    return a.tolist(), b.tolist()

# Pre-compute optimal coefficients
_a_opt, _b_opt = _optimize()

def coefficients(stages):
    """Return optimized 12-stage product formula coefficients."""
    a = list(_a_opt)
    b = list(_b_opt)
    
    # Pad or truncate to match stages if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
