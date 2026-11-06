import numpy as np
import time

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def tensor_product(matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result
    
    def site_op(op, site):
        ops = [ident] * N
        ops[site] = op
        return tensor_product(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        A += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        A += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        A += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        A -= (h / 2.0) * site_op(sz, j)
    
    A = (A + A.conj().T) / 2.0

    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        B += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        B += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        B += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        B -= (h / 2.0) * site_op(sz, j)
    
    B = (B + B.conj().T) / 2.0

    return A, B

# Pre-compute Hamiltonians and eigen-decompositions
_A, _B = _build_hamiltonians()
_H = _A + _B
_wA, _VA = np.linalg.eigh(_A)
_wB, _VB = np.linalg.eigh(_B)
_wH, _VH = np.linalg.eigh(_H)
_T = [0.25, 0.5, 1.0]

def _expm_hermitian_scaled(w, V, scale):
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

_exact_U = [_expm_hermitian_scaled(_wH, _VH, t) for t in _T]

def _compute_error_fast(a, b):
    """Compute mean spectral norm error. Uses precomputed eigendecomps."""
    errors = []
    dim = 64
    I = np.eye(dim, dtype=complex)
    for t, U_exact in zip(_T, _exact_U):
        U = I.copy()
        for k in range(12):
            bk = b[k]
            if abs(bk) > 1e-15:
                U = _expm_hermitian_scaled(_wB, _VB, bk * t) @ U
            ak = a[k]
            if abs(ak) > 1e-15:
                U = _expm_hermitian_scaled(_wA, _VA, ak * t) @ U
        errors.append(np.linalg.norm(U - U_exact, 2))
    return float(np.mean(errors))

def _is_valid(a, b):
    if len(a) != 12 or len(b) != 12:
        return False
    for x in list(a) + list(b):
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return False
    return True

def _make_symmetric(sym_a, sym_b):
    """Expand 6-element symmetric halves to 12-element full lists, normalized."""
    if np.any(~np.isfinite(sym_a)) or np.any(~np.isfinite(sym_b)):
        return None, None
    sa = np.sum(sym_a)
    sb = np.sum(sym_b)
    if abs(sa) < 1e-15 or abs(sb) < 1e-15:
        return None, None
    na = sym_a * (0.5 / sa)
    nb = sym_b * (0.5 / sb)
    if np.any(np.abs(na) > 50) or np.any(np.abs(nb) > 50):
        return None, None
    a = np.zeros(12)
    b = np.zeros(12)
    for i in range(6):
        a[i] = na[i]
        a[11-i] = na[i]
        b[i] = nb[i]
        b[11-i] = nb[i]
    return a, b

def _nelder_mead(objective, x0, bounds, max_iter=500, time_limit=80.0):
    """Nelder-Mead simplex method with bounds, time-limited."""
    start = time.time()
    n = len(x0)
    x = np.array([float(np.clip(x0[i], bounds[i][0], bounds[i][1])) for i in range(n)])
    
    # Initial simplex
    simplex = [x.copy()]
    fvals = [objective(x)]
    step = 0.1
    for i in range(n):
        xi = x.copy()
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)
        fvals.append(objective(xi))
    
    best_idx = int(np.argmin(fvals))
    best_x = simplex[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        if (it % 10 == 0) and (time.time() - start > time_limit * 0.9):
            break
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[n])
        xr = np.array([np.clip(xr[i], bounds[i][0], bounds[i][1]) for i in range(n)])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[n])
            xe = np.array([np.clip(xe[i], bounds[i][0], bounds[i][1]) for i in range(n)])
            fe = objective(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n] = fe
            else:
                simplex[n] = xr
                fvals[n] = fr
        elif fr < fvals[n-1]:
            simplex[n] = xr
            fvals[n] = fr
        else:
            # Contraction
            rho = 0.5
            if fr < fvals[n-1]:
                xc = centroid + rho * (simplex[n] - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                xc = centroid - rho * (simplex[n] - centroid)
                xc = np.array([np.clip(xc[i], bounds[i][0], bounds[i][1]) for i in range(n)])
                fc = objective(xc)
                if fc < fvals[n-1]:
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        # Track best
        bi = int(np.argmin(fvals))
        if fvals[bi] < best_f:
            best_f = fvals[bi]
            best_x = simplex[bi].copy()
    
    return best_x, best_f

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients.
    Uses Nelder-Mead on symmetric 12-dim parameterization with strict time limits.
    """
    start_time = time.time()
    HARD_DEADLINE = start_time + 130  # 20s safety margin before 150s kill
    
    # Fallback: uniform 12-stage (second-order accurate)
    fallback_a = [1.0/12.0] * 12
    fallback_b = [1.0/12.0] * 12
    
    # Try a few known-good symmetric starting points
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(([1.0/12.0]*12, [1.0/12.0]*12))
    
    # Candidate 2: Strang-like (more weight on ends for b)
    sa = [1.0/12.0]*12
    sb = [1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0]
    # Normalize sums
    ssa = sum(sa)
    ssb = sum(sb)
    sa = [x/ssa for x in sa]
    sb = [x/ssb for x in sb]
    candidates.append((sa, sb))
    
    # Candidate 3: 4th-order Suzuki-inspired symmetric
    # For 4th order: a = [1/2, 1/2, -1/2, 1/2, 1/2, -1/2, ...] pattern scaled
    # Let's use a smoother 4th-order-like distribution
    sa3 = [0.15, 0.20, 0.15, 0.15, 0.20, 0.15, 0.15, 0.20, 0.15, 0.15, 0.20, 0.15]
    sb3 = [0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10]
    ssa3 = sum(sa3)
    ssb3 = sum(sb3)
    sa3 = [x/ssa3 for x in sa3]
    sb3 = [x/ssb3 for x in sb3]
    candidates.append((sa3, sb3))
    
    # Evaluate all candidates, pick best
    best_a = list(fallback_a)
    best_b = list(fallback_b)
    best_err = float('inf')
    
    for ca, cb in candidates:
        if _is_valid(ca, cb):
            err = _compute_error_fast(np.array(ca), np.array(cb))
            if err < best_err:
                best_err = err
                best_a = list(ca)
                best_b = list(cb)
    
    # If we have time, run Nelder-Mead on symmetric parameters
    elapsed = time.time() - start_time
    remaining = HARD_DEADLINE - time.time()
    
    if remaining > 15:
        # Convert best to symmetric form for optimization
        # We'll optimize over 12 free parameters (not enforcing symmetry in the optimizer,
        # but starting from a symmetric point and using bounds)
        
        # Actually, let's optimize the full 12+12=24 params but with constraints
        # Better: use the symmetric representation (6+6=12 params)
        
        # Extract symmetric halves from best
        sym_a_init = np.array([(best_a[i] + best_a[11-i]) / 2.0 for i in range(6)])
        sym_b_init = np.array([(best_b[i] + best_b[11-i]) / 2.0 for i in range(6)])
        
        # Normalize so that when expanded, sums are 1
        # _make_symmetric handles normalization, so just use raw values
        
        x0 = np.concatenate([sym_a_init, sym_b_init])
        bounds = [(-5.0, 5.0)] * 12
        
        def objective(params):
            sa = params[:6]
            sb = params[6:12]
            a, b = _make_symmetric(sa, sb)
            if a is None:
                return 1e10
            if time.time() > HARD_DEADLINE:
                return 1e10
            return _compute_error_fast(a, b)
        
        try:
            opt_x, opt_f = _nelder_mead(objective, x0, bounds, max_iter=300, time_limit=min(remaining - 10, 100.0))
            
            if opt_f < best_err:
                sa_opt = opt_x[:6]
                sb_opt = opt_x[6:12]
                a_opt, b_opt = _make_symmetric(sa_opt, sb_opt)
                if a_opt is not None and _is_valid(a_opt, b_opt):
                    best_a = list(a_opt)
                    best_b = list(b_opt)
                    best_err = opt_f
        except Exception:
            pass
    
    # Final validation and return
    if not _is_valid(best_a, best_b):
        best_a = [1.0/12.0] * 12
        best_b = [1.0/12.0] * 12
    
    return list(best_a), list(best_b)

# EVOLVE-BLOCK-END
