import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

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

def _get_AB():
    global _A, _B, _A_eigvals, _A_eigvecs, _B_eigvals, _B_eigvecs
    if _A is None:
        _A, _B = _build_hamiltonians()
        _A_eigvals, _A_eigvecs = np.linalg.eigh(_A)
        _B_eigvals, _B_eigvecs = np.linalg.eigh(_B)
    return _A, _B

def _expm_A(t):
    """exp(-i A t) using precomputed eigendecomposition."""
    phase = np.exp(-1j * _A_eigvals * t)
    return (_A_eigvecs * phase) @ _A_eigvecs.conj().T

def _expm_B(t):
    """exp(-i B t) using precomputed eigendecomposition."""
    phase = np.exp(-1j * _B_eigvals * t)
    return (_B_eigvecs * phase) @ _B_eigvecs.conj().T

def _expm_AB(t):
    """exp(-i (A+B) t) for the exact propagator."""
    H = _A + _B
    eigvals, eigvecs = np.linalg.eigh(H)
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def _clamp_to_box(x, lo=-50.0, hi=50.0):
    return np.clip(x, lo, hi)

def coefficients(stages):
    """Optimize 24 coefficients using L-BFGS-B with precomputed eigendecompositions."""
    if time.time() > _DEADLINE - 20:
        sub = stages // 2
        a = [0.5 / sub] * (2 * sub)
        b = []
        for _ in range(sub):
            b += [1.0 / sub, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        return a[:stages], b[:stages]

    _get_AB()
    t_list = [0.25, 0.5, 1.0]
    dim = 64

    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _expm_AB(t)

    # Precompute all needed matrix exponentials for a grid of coefficient values
    # We'll cache exp(-i*A*c*t) and exp(-i*B*c*t) for various c values
    # Actually, let's just compute on the fly but with fast eig-based method

    def U_of(a, b, t):
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            Eb = _expm_B(b[k] * t)
            Ea = _expm_A(a[k] * t)
            U = Eb @ Ea @ U
        return U

    def obj(theta):
        if time.time() > _DEADLINE - 5:
            return 1e10
        a_free = theta[:11]
        b_free = theta[11:22]
        a = np.empty(12); a[:11] = a_free; a[11] = 1.0 - np.sum(a_free)
        b = np.empty(12); b[:11] = b_free; b[11] = 1.0 - np.sum(b_free)
        # Enforce box constraints softly (L-BFGS-B handles them, but be safe)
        a = _clamp_to_box(a)
        b = _clamp_to_box(b)
        total = 0.0
        for t in t_list:
            U = U_of(a, b, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Use time-reversal symmetric structure to reduce parameters:
    # a_k = a_{13-k}, b_k = b_{13-k}
    # This gives 6 free a's and 6 free b's = 12 parameters total
    # Plus the sum constraint: sum(a)=1, sum(b)=1
    # So 5 free a's + 5 free b's = 10 parameters

    def obj_symmetric(theta):
        if time.time() > _DEADLINE - 5:
            return 1e10
        a_half = theta[:5]  # a[0..4], a[5] determined by symmetry
        b_half = theta[5:10]  # b[0..4], b[5] determined by symmetry
        # Build full symmetric arrays: a[k] = a[11-k]
        a = np.zeros(12)
        for k in range(6):
            a[k] = a_half[k]
            a[11 - k] = a_half[k]
        # sum(a) = 2*sum(a_half) = 1 => sum(a_half) = 0.5
        # But we have 6 values summing to 0.5... let's use 5 free + 1 determined
        a = np.zeros(12)
        for k in range(5):
            a[k] = a_half[k]
            a[11 - k] = a_half[k]
        a[5] = 0.5 - np.sum(a_half)
        a[6] = a[5]  # symmetry: a[5] = a[6] since 11-5=6
        
        b = np.zeros(12)
        for k in range(5):
            b[k] = b_half[k]
            b[11 - k] = b_half[k]
        b[5] = 0.5 - np.sum(b_half)
        b[6] = b[5]
        
        a = _clamp_to_box(a)
        b = _clamp_to_box(b)
        
        total = 0.0
        for t in t_list:
            U = U_of(a, b, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Initial guess: uniform Strang-like
    theta0 = np.zeros(10)
    for i in range(5):
        theta0[i] = 0.5 / 6.0  # each pair contributes 2*(1/12)=1/6, 5 pairs = 5/6, middle = 1/6... 
        # Actually for 12 stages with symmetry: 6 pairs, each pair value v, sum = 2*6*v = 1 => v = 1/12
        theta0[i] = 1.0 / 12.0
        theta0[5 + i] = 1.0 / 12.0
    # But a[5] = 0.5 - sum(a_half) = 0.5 - 5*(1/12) = 0.5 - 5/12 = 1/12. Good.

    best_theta = theta0.copy()
    best_val = obj_symmetric(theta0)

    # Try L-BFGS-B with box constraints
    try:
        bounds = [(-50.0, 50.0)] * 10
        result = scipy_minimize(
            obj_symmetric, theta0, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_theta = result.x
    except Exception:
        pass

    # Also try a second run from a different initial point if time permits
    if time.time() < _DEADLINE - 30:
        try:
            # Try a slightly perturbed starting point
            theta1 = theta0.copy() + np.random.normal(0, 0.01, 10)
            result2 = scipy_minimize(
                obj_symmetric, theta1, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_theta = result2.x
        except Exception:
            pass

    # Extract final coefficients from symmetric parameterization
    a_half = best_theta[:5]
    b_half = best_theta[5:10]
    a = np.zeros(12)
    for k in range(5):
        a[k] = a_half[k]
        a[11 - k] = a_half[k]
    a[5] = 0.5 - np.sum(a_half)
    a[6] = a[5]
    
    b = np.zeros(12)
    for k in range(5):
        b[k] = b_half[k]
        b[11 - k] = b_half[k]
    b[5] = 0.5 - np.sum(b_half)
    b[6] = b[5]
    
    a = _clamp_to_box(a).tolist()
    b = _clamp_to_box(b).tolist()

    # Verify sum constraints
    if abs(sum(a) - 1.0) > 1e-6:
        a = [x / sum(a) for x in a]
    if abs(sum(b) - 1.0) > 1e-6:
        b = [x / sum(b) for x in b]

    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
