import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half uniform field) and B (even bonds + half uniform field)
    for a 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op, N):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz, N)
        B += 0.5 * field * op_on_site(j, sz, N)
    
    return A, B

# Pre-compute matrices and decompositions to speed up evaluation
_A, _B = _build_hamiltonians()
_H = _A + _B

# Eigendecompositions for A, B, and H
# A and B are Hermitian, so we can use eigh for better stability and real eigenvalues
evals_A, evecs_A = np.linalg.eigh(_A)
evals_B, evecs_B = np.linalg.eigh(_B)
evals_H, evecs_H = np.linalg.eigh(_H)

# Inverse of eigenvectors (conjugate transpose for unitary)
inv_evecs_A = np.linalg.inv(evecs_A)
inv_evecs_B = np.linalg.inv(evecs_B)
inv_evecs_H = np.linalg.inv(evecs_H)

# Exact propagator for the three time steps
_T_VALS = [0.25, 0.5, 1.0]
_EXACT_PROPS = []
for t in _T_VALS:
    # exp(-i t H)
    phase = np.exp(-1j * t * evals_H)
    exact_prop = evecs_H @ np.diag(phase) @ inv_evecs_H
    _EXACT_PROPS.append(exact_prop)

def _eval_product_formula(a, b, t):
    """
    Compute U(t) = prod_{k=0}^{11} exp(-i b_k t B) exp(-i a_k t A)
    a, b are arrays of length 12.
    """
    U = np.eye(64, dtype=complex)
    
    # Pre-compute exponentials for this t? No, a_k and b_k vary.
    # But we can compute exp(-i a_k t A) using the pre-computed eigh.
    
    for k in range(12):
        # exp(-i a_k t A)
        phase_a = np.exp(-1j * a[k] * t * evals_A)
        exp_A_k = evecs_A @ np.diag(phase_a) @ inv_evecs_A
        
        # exp(-i b_k t B)
        phase_b = np.exp(-1j * b[k] * t * evals_B)
        exp_B_k = evecs_B @ np.diag(phase_b) @ inv_evecs_B
        
        # U = exp_B_k @ exp_A_k @ U
        U = exp_B_k @ (exp_A_k @ U)
        
    return U

def _objective(params):
    """
    params: 10-element vector.
    First 5 are free params for a (x1..x5), 6th a is derived.
    Next 5 are free params for b (y1..y5), 6th b is derived.
    
    Symmetric pattern:
    a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    
    Constraint: sum(a) = 1 => 2*(x1+...+x6) = 1 => x6 = 0.5 - (x1+...+x5)
    Constraint: sum(b) = 1 => 2*(y1+...+y6) = 1 => y6 = 0.5 - (y1+...+y5)
    """
    x = params[:5]
    y = params[5:]
    
    x6 = 0.5 - np.sum(x)
    y6 = 0.5 - np.sum(y)
    
    a = np.array([x[0], x[1], x[2], x[3], x[4], x6, x6, x[4], x[3], x[2], x[1], x[0]])
    b = np.array([y[0], y[1], y[2], y[3], y[4], y6, y6, y[4], y[3], y[2], y[1], y[0]])
    
    # Check bounds [-50, 50]
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
        
    errors = []
    for i, t in enumerate(_T_VALS):
        U_approx = _eval_product_formula(a, b, t)
        U_exact = _EXACT_PROPS[i]
        diff = U_approx - U_exact
        # Spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        err = s[0] if len(s) > 0 else 0.0
        errors.append(err)
        
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Equal weights
    # a_k = 1/12, b_k = 1/12
    # Symmetric representation:
    # 2 * sum(x) = 1 => sum(x) = 0.5
    # x1..x6 = 1/12
    # x6 = 1/12
    # sum(x1..x5) = 5/12 = 0.4166...
    # x_i = 1/12 for i=1..5
    x_init = np.array([1.0/12.0] * 5)
    y_init = np.array([1.0/12.0] * 5)
    x0 = np.concatenate([x_init, y_init])
    
    # Try a few starting points
    best_x = x0
    best_f = _objective(x0)
    
    # Nelder-Mead is robust for small dimensions
    # L-BFGS-B might be faster if gradient is available, but we don't have analytic gradient.
    # Let's use Nelder-Mead with tight tolerances.
    
    res = minimize(
        _objective, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
    )
    
    if time.time() > deadline:
        # Return best found so far
        pass
    else:
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    
    # Try a second run from a slightly different start to avoid local minima
    if time.time() < deadline - 20.0:
        # Perturb initial guess
        x0_2 = x0 + np.random.normal(0, 0.01, 10)
        res2 = minimize(
            _objective, 
            x0_2, 
            method='Nelder-Mead', 
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x

    # Construct final a and b
    x = best_x[:5]
    y = best_x[5:]
    x6 = 0.5 - np.sum(x)
    y6 = 0.5 - np.sum(y)
    
    a = [x[0], x[1], x[2], x[3], x[4], x6, x6, x[4], x[3], x[2], x[1], x[0]]
    b = [y[0], y[1], y[2], y[3], y[4], y6, y6, y[4], y[3], y[2], y[1], y[0]]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]
    
    # Normalize to ensure sum = 1 (handle floating point errors)
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-10:
        a = [v / sum_a for v in a]
    if abs(sum_b) > 1e-10:
        b = [v / sum_b for v in b]
        
    return a, b

# EVOLVE-BLOCK-END
