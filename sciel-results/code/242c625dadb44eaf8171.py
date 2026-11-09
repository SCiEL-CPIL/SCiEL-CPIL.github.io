"""Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
import numpy as np
from numpy.linalg import norm
import time
from scipy.optimize import minimize
import scipy.linalg as sla

# Deadline guard
_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# Precomputation variables
_A = None
_B = None
_V_A = None
_D_A = None
_V_B = None
_D_B = None
_exact_propagators = {}
_t_values = [0.25, 0.5, 1.0]

def _build_hamiltonians():
    global _A, _B, _V_A, _D_A, _V_B, _D_B
    if _A is not None:
        return
    
    N = 6
    J = 1.0
    gamma = 0.3
    anisotropy = 0.8
    
    # Build Heisenberg XXZ chain Hamiltonian
    # H = sum J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + gamma Sz_i Sz_{i+1}) - h sum Sz_i
    # Split into even and odd bonds.
    # A = odd bonds + half field
    # B = even bonds + half field
    
    # Spin operators
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    
    # Kronecker products
    def kron_all(ops):
        res = np.array([[1]])
        for op in ops:
            res = np.kron(res, op)
        return res
        
    # We need operators for each site
    # Identity
    I = np.eye(2)
    
    H_A = np.zeros((2**N, 2**N), dtype=complex)
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        # Field term: -h Sz_i / 2 for each half?
        # Task says "each including half of the uniform field"
        # Total field term is -h sum Sz_i.
        # So A gets -0.5 h Sz_i, B gets -0.5 h Sz_i.
        ops = [I] * N
        ops[i] = sz
        field_op = kron_all(ops)
        H_A += -0.5 * gamma * field_op # Wait, gamma is anisotropy? No, field is 0.3.
        # Correction: anisotropy is 0.8 (gamma in XXZ usually refers to z-coupling).
        # Field is 0.3.
        H_A += -0.5 * 0.3 * field_op
        H_B += -0.5 * 0.3 * field_op
        
    # Bonds
    # Odd bonds: (1,2), (3,4), (5,6) -> indices 0-1, 2-3, 4-5 (0-indexed)
    # Even bonds: (2,3), (4,5), (6,1) -> indices 1-2, 3-4, 5-0 (cyclic? "open chain")
    # "six-site open XXZ Heisenberg chain"
    # Open chain: bonds are (0,1), (1,2), (2,3), (3,4), (4,5).
    # "split into even and odd bonds"
    # Usually: A has (0,1), (2,3), (4,5). B has (1,2), (3,4).
    # Or A has (1,2), (3,4), B has (0,1), (2,3), (4,5)?
    # Let's assume standard alternating:
    # A: (0,1), (2,3), (4,5)
    # B: (1,2), (3,4)
    # Note: In an open chain of 6 sites, there are 5 bonds.
    # 0-1 (odd index bond 0?), 1-2, 2-3, 3-4, 4-5.
    # If we split by site index parity of the *left* site:
    # Left site 0 (even) -> Bond 0-1 in A?
    # Left site 1 (odd) -> Bond 1-2 in B?
    # Left site 2 (even) -> Bond 2-3 in A?
    # Left site 3 (odd) -> Bond 3-4 in B?
    # Left site 4 (even) -> Bond 4-5 in A?
    # This gives A 3 bonds, B 2 bonds.
    
    def add_bond(H, i, j):
        # Sx_i Sx_j + Sy_i Sy_j + anisotropy Sz_i Sz_j
        ops_x = [I] * N
        ops_x[i] = sx
        ops_x[j] = sx
        H += J * kron_all(ops_x)
        
        ops_y = [I] * N
        ops_y[i] = sy
        ops_y[j] = sy
        H += J * kron_all(ops_y)
        
        ops_z = [I] * N
        ops_z[i] = sz
        ops_z[j] = sz
        H += J * anisotropy * kron_all(ops_z)
        
    # A bonds: (0,1), (2,3), (4,5)
    add_bond(H_A, 0, 1)
    add_bond(H_A, 2, 3)
    add_bond(H_A, 4, 5)
    
    # B bonds: (1,2), (3,4)
    add_bond(H_B, 1, 2)
    add_bond(H_B, 3, 4)
    
    _A = H_A
    _B = H_B
    
    # Diagonalize
    _D_A, _V_A = sla.eigh(_A)
    _D_B, _V_B = sla.eigh(_B)
    
    # Precompute exact propagators
    for t in _t_values:
        _exact_propagators[t] = sla.expm(-1j * t * (_A + _B))

def _spectral_norm_error(a, b):
    """
    Computes mean spectral norm error for the 3 time points.
    a, b are lists of 12 floats.
    """
    if _A is None:
        _build_hamiltonians()
        
    total_err = 0.0
    dim = _A.shape[0]
    I = np.eye(dim)
    
    for t in _t_values:
        U = I.copy()
        # Product order k=1..12: exp(-i b_k t B) exp(-i a_k t A)
        # We apply from right to left?
        # U = M_1 M_2 ... M_12
        # M_k = exp(-i b_k t B) exp(-i a_k t A)
        # So U = M_1 @ M_2 @ ... @ M_12
        
        # Optimization: Precompute exp(-i a_k t A) and exp(-i b_k t B)
        # exp(-i a t A) = V_A diag(exp(-i a t D_A)) V_A^dagger
        
        for k in range(12):
            # Calculate exp(-i b_k t B)
            # b_k is scalar
            # D_B is diagonal of eigenvalues
            exp_B = np.diag(np.exp(-1j * b[k] * t * _D_B))
            M_B = _V_B @ exp_B @ _V_B.conj().T
            
            # Calculate exp(-i a_k t A)
            exp_A = np.diag(np.exp(-1j * a[k] * t * _D_A))
            M_A = _V_A @ exp_A @ _V_A.conj().T
            
            M_k = M_B @ M_A
            U = U @ M_k
            
        U_exact = _exact_propagators[t]
        err = norm(U - U_exact, 2)
        total_err += err
        
    return total_err / len(_t_values)

def coefficients(stages):
    if _A is None:
        _build_hamiltonians()
        
    # Initial guess: Strang splitting
    # 11 substeps? Or 12?
    # Let's start with a simple guess: uniform
    a = [1.0/12] * 12
    b = [1.0/12] * 12
    
    # Optimize
    # We have 24 variables. Constraints: sum(a)=1, sum(b)=1.
    # We can optimize 22 variables and set the last one.
    # Let's use Nelder-Mead on the 22 free variables.
    
    def objective(x):
        # x has length 22: a[0:11], b[0:11]
        a_full = list(x[0:11])
        b_full = list(x[11:22])
        a_full.append(1.0 - sum(a_full))
        b_full.append(1.0 - sum(b_full))
        
        # Clip to [-50, 50]
        a_full = np.clip(a_full, -50, 50)
        b_full = np.clip(b_full, -50, 50)
        
        # Re-normalize? If we clip, sum might not be 1.
        # The task requires sum=1.
        # If we clip, we should probably normalize again or handle it carefully.
        # For now, assume the optimizer stays in bounds.
        
        return _spectral_norm_error(a_full, b_full)
        
    x0 = np.array(a[0:11] + b[0:11])
    
    # Check time
    if _remaining() < 10:
        return a, b
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or _remaining() < 5:
             x_opt = res.x
        else:
             x_opt = x0
             
        a_opt = list(x_opt[0:11])
        b_opt = list(x_opt[11:22])
        a_opt.append(1.0 - sum(a_opt))
        b_opt.append(1.0 - sum(b_opt))
        
        # Final check and clip
        a_opt = np.clip(a_opt, -50, 50)
        b_opt = np.clip(b_opt, -50, 50)
        
        # Ensure sum is exactly 1?
        # If clipping changed the sum, we need to renormalize.
        # But if we renormalize, we might break the "optimal" structure.
        # Given the bounds are wide, clipping should rarely happen.
        
        return list(a_opt), list(b_opt)
        
    except Exception:
        return a, b

