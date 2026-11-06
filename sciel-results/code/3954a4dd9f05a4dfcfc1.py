import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds 2-3,4-5 + half field) and B (even bonds 1-2,3-4,5-6 + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        """op (2x2) acting on site (0-indexed)."""
        ops = []
        for s in range(n):
            ops.append(op if s == site else I)
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
_A_eigvals = None
_A_eigvecs = None
_B_eigvals = None
_B_eigvecs = None
_AB_eigvals = None
_AB_eigvecs = None

def _get_AB():
    global _A, _B, _A_eigvals, _A_eigvecs, _B_eigvals, _B_eigvecs, _AB_eigvals, _AB_eigvecs
    if _A is None:
        _A, _B = _build_hamiltonians()
        _A_eigvals, _A_eigvecs = np.linalg.eigh(_A)
        _B_eigvals, _B_eigvecs = np.linalg.eigh(_B)
        _AB_eigvals, _AB_eigvecs = np.linalg.eigh(_A + _B)
    return _A, _B

def _expm_from_eigh(eigvals, eigvecs, t):
    """Compute exp(-i H t) using precomputed eigen-decomposition."""
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def _clamp_to_box(x, lo=-50.0, hi=50.0):
    return np.clip(x, lo, hi)

def coefficients(stages):
    """Optimize symmetric 12-stage coefficients using SciPy minimize."""
    if time.time() > _DEADLINE - 20:
        sub = stages // 2
        a = [0.5 / sub] * (2 * sub)
        b = []
        for _ in range(sub):
            b += [1.0 / sub, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        return a[:stages], b[:stages]

    A, B = _get_AB()
    
    # Chebyshev grid: use more time points to approximate the uniform norm
    t_grid = np.linspace(0.05, 1.0, 8)
    
    # Precompute exact propagators on the grid
    exact_grid = {}
    for t in t_grid:
        exact_grid[t] = _expm_from_eigh(_AB_eigvals, _AB_eigvecs, t)
    
    def U_of(a, b, t):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            Eb = _expm_from_eigh(_B_eigvals, _B_eigvecs, b[k] * t)
            Ea = _expm_from_eigh(_A_eigvals, _A_eigvecs, a[k] * t)
            U = Eb @ Ea @ U
        return U

    def obj_sym(theta):
        """
        theta: 10 parameters (5 for a, 5 for b)
        Symmetric 12-stage: a = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
        Sum(a) = 2 * sum(a0..a5) = 1 => sum(a0..a5) = 0.5
        So we fix a5 = 0.5 - sum(a0..a4).
        Same for b.
        """
        a_free = theta[:5]
        b_free = theta[5:10]
        
        a5 = 0.5 - np.sum(a_free)
        b5 = 0.5 - np.sum(b_free)
        
        a_half = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a5])
        b_half = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b5])
        
        a = np.concatenate([a_half, a_half[::-1]])
        b = np.concatenate([b_half, b_half[::-1]])
        
        a = _clamp_to_box(a)
        b = _clamp_to_box(b)
        
        # Re-normalize after clamping? No, clamping might break sum=1.
        # If clamping occurs, we should probably project back or just accept the error.
        # Given the bounds are wide [-50, 50] and typical coefficients are small,
        # clamping is unlikely to be active unless the optimizer runs away.
        
        max_err = 0.0
        for t in t_grid:
            U = U_of(a, b, t)
            err = U - exact_grid[t]
            err_norm = np.linalg.norm(err, 2)
            if err_norm > max_err:
                max_err = err_norm
        return max_err

    # Initial guess: equal weights (Strang-like)
    # a_half sum = 0.5 => each 0.5/6
    theta0 = np.full(10, 0.5 / 6.0)
    
    # Try to improve with Powell
    best_val = obj_sym(theta0)
    best_theta = theta0.copy()
    
    try:
        res = minimize(obj_sym, theta0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x
    except Exception:
        pass

    # Extract final coefficients
    a_free = best_theta[:5]
    b_free = best_theta[5:10]
    
    a5 = 0.5 - np.sum(a_free)
    b5 = 0.5 - np.sum(b_free)
    
    a_half = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a5])
    b_half = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b5])
    
    a = np.concatenate([a_half, a_half[::-1]])
    b = np.concatenate([b_half, b_half[::-1]])
    
    a = _clamp_to_box(a).tolist()
    b = _clamp_to_box(b).tolist()

    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
