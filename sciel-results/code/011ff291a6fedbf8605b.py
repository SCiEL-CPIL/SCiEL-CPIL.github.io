import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds 0-1,2-3,4-5 + half field) and B (even bonds 1-2,3-4 + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)

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

    def site_op(op, site):
        ops = []
        for s in range(n):
            ops.append(op if s == site else I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    # XXZ bond: sx⊗sx + sy⊗sy + delta * sz⊗sz
    def xxz_bond(s1, s2):
        return bond_op(sx, sx, s1, s2) + bond_op(sy, sy, s1, s2) + delta * bond_op(sz, sz, s1, s2)

    # A: odd bonds (0-1, 2-3, 4-5) + half field on all sites
    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(0, 1)
    A += xxz_bond(2, 3)
    A += xxz_bond(4, 5)
    for s in range(n):
        A -= (field / 2.0) * site_op(sz, s)
    A = (A + A.conj().T) / 2.0

    # B: even bonds (1-2, 3-4) + half field on all sites
    B = np.zeros((dim, dim), dtype=complex)
    B += xxz_bond(1, 2)
    B += xxz_bond(3, 4)
    for s in range(n):
        B -= (field / 2.0) * site_op(sz, s)
    B = (B + B.conj().T) / 2.0

    return A, B

# Pre-compute Hamiltonians and eigendecompositions
_A, _B = _build_hamiltonians()
_H = _A + _B

_wA, _VA = np.linalg.eigh(_A)
_wB, _VB = np.linalg.eigh(_B)
_wH, _VH = np.linalg.eigh(_H)

_T = [0.25, 0.5, 1.0]
_dim = 64

def _expm_hermitian(w, V, scale):
    """Compute exp(-i * scale * H) given eigenvalues w and eigenvectors V."""
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

# Pre-compute exact propagators
_exact_U = [_expm_hermitian(_wH, _VH, t) for t in _T]

def _compute_error(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(_dim, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U_B = _expm_hermitian(_wB, _VB, b[k] * t)
                U = U_B @ U
            if a[k] != 0.0:
                U_A = _expm_hermitian(_wA, _VA, a[k] * t)
                U = U_A @ U
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _is_valid(a, b):
    """Check if coefficients are valid."""
    if len(a) != 12 or len(b) != 12:
        return False
    for x in list(a) + list(b):
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return False
    return True

def _nelder_mead(f, x0, max_iter=500, tol=1e-8, deadline=None):
    """Nelder-Mead optimization with time limit."""
    start_time = time.time()
    n = len(x0)
    
    # Initial simplex with small perturbations
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05
        x_new[i] += step
        simplex.append(x_new)
    
    f_values = [f(simplex[i]) for i in range(n + 1)]
    best_x = simplex[f_values.index(min(f_values))].copy()
    best_f = min(f_values)
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if time.time() - start_time > 130:
            break
        
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        if abs(f_values[-1] - f_values[0]) < tol:
            break
        
        centroid = np.mean(simplex[:n], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[n])
        f_r = f(x_r)
        
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[n] = x_e
                f_values[n] = f_e
            else:
                simplex[n] = x_r
                f_values[n] = f_r
        elif f_r < f_values[n-1]:
            simplex[n] = x_r
            f_values[n] = f_r
        else:
            x_c = centroid + rho * (simplex[n] - centroid)
            f_c = f(x_c)
            if f_c < f_values[n]:
                simplex[n] = x_c
                f_values[n] = f_c
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = f(simplex[i])
        
        current_best_idx = f_values.index(min(f_values))
        if f_values[current_best_idx] < best_f:
            best_f = f_values[current_best_idx]
            best_x = simplex[current_best_idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    """Find optimal 12-stage product formula coefficients."""
    if time.time() > _DEADLINE - 15:
        # Not enough time; return a sane Strang-like default
        sub = max(1, stages // 2)
        a = [0.5 / sub] * (2 * sub)
        b = []
        for _ in range(sub):
            b += [1.0 / sub, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        return a[:stages], b[:stages]
    
    # Fallback: Simple Strang splitting
    fallback_a = [1.0/12.0] * 12
    fallback_b = [0.0] * 12
    for i in range(0, 12, 2):
        fallback_b[i] = 1.0/6.0
    
    best_params = (fallback_a, fallback_b)
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Objective function: 22 free parameters (11 a's + 11 b's, last determined by sum=1)
    def objective(theta):
        a_free = theta[:11]
        b_free = theta[11:22]
        a = np.empty(12)
        b = np.empty(12)
        a[:11] = a_free
        a[11] = 1.0 - np.sum(a_free)
        b[:11] = b_free
        b[11] = 1.0 - np.sum(b_free)
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6
        
        return _compute_error(a, b)
    
    # Initial guess: uniform Strang-like
    theta0 = np.zeros(22)
    for i in range(11):
        theta0[i] = 1.0 / 12.0
        theta0[11 + i] = 1.0 / 12.0
    # Last a and b will be 1 - 11*(1/12) = 1/12, good
    
    # Run Nelder-Mead
    try:
        x_best, f_best = _nelder_mead(objective, theta0, max_iter=800, tol=1e-7, deadline=_DEADLINE)
        
        a_free = x_best[:11]
        b_free = x_best[11:22]
        a_best = np.empty(12)
        b_best = np.empty(12)
        a_best[:11] = a_free
        a_best[11] = 1.0 - np.sum(a_free)
        b_best[:11] = b_free
        b_best[11] = 1.0 - np.sum(b_free)
        
        # Clamp to valid range
        a_best = np.clip(a_best, -50, 50)
        b_best = np.clip(b_best, -50, 50)
        
        # Re-normalize after clamping
        sum_a = np.sum(a_best)
        sum_b = np.sum(b_best)
        if abs(sum_a) > 1e-12:
            a_best = a_best / sum_a
        if abs(sum_b) > 1e-12:
            b_best = b_best / sum_b
        
        if _is_valid(list(a_best), list(b_best)):
            err = _compute_error(a_best, b_best)
            if err < best_error:
                best_error = err
                best_params = (list(a_best), list(b_best))
    except Exception:
        pass
    
    # Try a second optimization run from a different starting point if time permits
    if time.time() < _DEADLINE - 30:
        try:
            # Try a symmetric 4th-order-like starting point
            theta1 = np.zeros(22)
            # Symmetric pattern for higher order
            c = 1.0 / (2.0 * (1.0 - 2.0 ** (-0.5)))  # ~1.707
            theta1[0] = c / 12
            theta1[11] = c / 12
            for i in range(1, 6):
                theta1[i] = (1.0 - c) / 6
                theta1[11 + i] = (1.0 - c) / 6
            theta1[6] = 1.0 - np.sum(theta1[:6])
            theta1[17] = 1.0 - np.sum(theta1[11:17])
            
            x_best2, f_best2 = _nelder_mead(objective, theta1, max_iter=400, tol=1e-7, deadline=_DEADLINE)
            
            a_free = x_best2[:11]
            b_free = x_best2[11:22]
            a_best = np.empty(12)
            b_best = np.empty(12)
            a_best[:11] = a_free
            a_best[11] = 1.0 - np.sum(a_free)
            b_best[:11] = b_free
            b_best[11] = 1.0 - np.sum(b_free)
            
            a_best = np.clip(a_best, -50, 50)
            b_best = np.clip(b_best, -50, 50)
            
            sum_a = np.sum(a_best)
            sum_b = np.sum(b_best)
            if abs(sum_a) > 1e-12:
                a_best = a_best / sum_a
            if abs(sum_b) > 1e-12:
                b_best = b_best / sum_b
            
            if _is_valid(list(a_best), list(b_best)):
                err = _compute_error(a_best, b_best)
                if err < best_error:
                    best_error = err
                    best_params = (list(a_best), list(b_best))
        except Exception:
            pass
    
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
    
    return fallback_a, fallback_b

# EVOLVE-BLOCK-END
