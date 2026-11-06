import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        ops = [op if s == site else I for s in range(n)]
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = []
        for s in range(n):
            if s == s1:
                ops.append(op1)
            elif s == s2:
                ops.append(op2)
            else:
                ops.append(I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    B = np.zeros((dim, dim), dtype=complex)
    B += xxz_bond(0, 1)
    B += xxz_bond(2, 3)
    B += xxz_bond(4, 5)
    for s in range(n):
        B += (field / 2.0) * site_op(sz, s)

    return A, B

_A = None
_B = None

def _get_AB():
    global _A, _B
    if _A is None:
        _A, _B = _build_hamiltonians()
    return _A, _B

def _matrix_exp(H, t):
    """Compute exp(-i H t) for Hermitian H."""
    if t == 0:
        return np.eye(H.shape[0], dtype=complex)
    eigvals, eigvecs = np.linalg.eigh(H)
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def coefficients(stages):
    """Optimize 24 coefficients to minimize mean spectral-norm error."""
    if time.time() > _DEADLINE - 20:
        # Fallback: uniform split
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a[:stages], b[:stages]

    A, B = _get_AB()
    t_list = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _matrix_exp(A + B, t)

    # Precompute eigendecompositions for A and B to speed up expm calls
    # exp(-i a t A) = V_A diag(exp(-i a t lambda_A)) V_A^dagger
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)

    def _expm_A(scale):
        # scale = a_k * t
        if scale == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * eigvals_A * scale)
        return (eigvecs_A * phase) @ eigvecs_A.conj().T

    def _expm_B(scale):
        # scale = b_k * t
        if scale == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * eigvals_B * scale)
        return (eigvecs_B * phase) @ eigvecs_B.conj().T

    # Objective function using symmetric parameterization
    # Free params: a1..a5, b1..b5 (10 params)
    # a6 = 0.5 - sum(a1..a5), a7=a6, a8=a5, ... a12=a1
    # b6 = 0.5 - sum(b1..b5), b7=b6, ... b12=b1
    
    def objective(theta):
        a_free = theta[:5]
        b_free = theta[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        # Construct full 12-length vectors
        a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
        b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
        
        # Check bounds (soft penalty or hard clip? L-BFGS-B handles bounds)
        # But we need to ensure the objective is valid.
        
        total_err = 0.0
        for t in t_list:
            U = np.eye(64, dtype=complex)
            for k in range(12):
                Eb = _expm_B(b[k] * t)
                Ea = _expm_A(a[k] * t)
                U = Eb @ Ea @ U
            err = U - exact[t]
            total_err += np.linalg.norm(err, 2)
        return total_err / len(t_list)

    # Initial guess: uniform
    # a_i = 1/12 for all i.
    # Symmetric: a1..a5 = 1/12, a6 = 0.5 - 5/12 = 1/12.
    theta0 = np.array([1.0/12.0] * 5 + [1.0/12.0] * 5)
    
    bounds = [(-50.0, 50.0)] * 10
    
    best_theta = theta0.copy()
    best_val = objective(theta0)
    
    try:
        # Use L-BFGS-B
        result = minimize(objective, theta0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_theta = result.x
    except Exception:
        pass

    # If we have time, try a second run from a different start or with different settings
    if time.time() < _DEADLINE - 10:
        try:
            # Perturb initial guess slightly or use a different method
            theta0_alt = theta0 + np.random.normal(0, 0.01, 10)
            result2 = minimize(objective, theta0_alt, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if result2.success and result2.fun < best_val:
                best_val = result2.fun
                best_theta = result2.x
        except Exception:
            pass

    # Extract coefficients
    a_free = best_theta[:5]
    b_free = best_theta[5:10]
    a6 = 0.5 - np.sum(a_free)
    b6 = 0.5 - np.sum(b_free)
    
    a = [a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]]
    b = [b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]]
    
    # Clamp to [-50, 50] just in case, though bounds should prevent this
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Ensure sums are exactly 1 by adjusting the last element slightly if needed
    # Due to floating point, sum might be 0.9999999
    sum_a = sum(a)
    a[11] += (1.0 - sum_a)
    sum_b = sum(b)
    b[11] += (1.0 - sum_b)
    
    # Re-clamp after adjustment
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]

    return a[:stages], b[:stages]
