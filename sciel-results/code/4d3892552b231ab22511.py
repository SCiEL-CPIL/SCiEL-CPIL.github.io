"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Structural change: Use a parameterized family of coefficients based on
known high-order splitting schemes (Suzuki-Trotter, Yoshida, etc.) and
optimize a small number of free parameters rather than all 24 coefficients.
This reduces the effective dimensionality and allows more effective search.
"""
import numpy as np
import time

_DEADLINE = 145.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j):
        ops_sz = [I] * N
        ops_sz[i] = sz
        ops_sz[j] = sz
        ops_sx = [I] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))
    
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

# Precompute eigen-decompositions (cached globally)
_eig_cache = None

def _get_eig():
    """Get cached eigen-decompositions of A, B, and H=A+B."""
    global _eig_cache
    if _eig_cache is None:
        A, B = _build_hamiltonians()
        H = A + B
        ev_A, vec_A = np.linalg.eigh(A)
        ev_B, vec_B = np.linalg.eigh(B)
        ev_H, vec_H = np.linalg.eigh(H)
        dim = A.shape[0]
        
        # Precompute exact propagators at target times
        exact_props = {}
        for t in [0.25, 0.5, 1.0]:
            exact_props[t] = (vec_H * np.exp(-1j * t * ev_H)) @ vec_H.conj().T
        
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
            'exact_props': exact_props,
            'vec_A_H': vec_A.conj().T,
            'vec_B_H': vec_B.conj().T,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    ev_A, vec_A = cache['ev_A'], cache['vec_A']
    ev_B, vec_B = cache['ev_B'], cache['vec_B']
    vec_A_H = cache['vec_A_H']
    vec_B_H = cache['vec_B_H']
    exact_props = cache['exact_props']
    dim = cache['dim']
    
    times = [0.25, 0.5, 1.0]
    errors = []
    
    for t in times:
        exact = exact_props[t]
        
        # Product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            exp_b = (vec_B * np.exp(-1j * b[k] * t * ev_B)) @ vec_B_H
            exp_a = (vec_A * np.exp(-1j * a[k] * t * ev_A)) @ vec_A_H
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    
    return np.mean(errors)

def _make_symmetric_4th_order(params):
    """Generate 12-stage coefficients from a symmetric 4th-order-like pattern.
    
    params: list of free parameters to tune
    Returns (a, b) each of length 12
    """
    # Start with a base pattern and apply symmetry
    # For 12 stages, we can use a pattern with mirror symmetry
    # a[0]=a[11], a[1]=a[10], ..., a[5]=a[6]
    # Same for b
    
    # Base 4th-order Suzuki pattern for 4 stages, scaled to 12
    # We'll use a parameterized family
    
    p = params
    
    # Create symmetric a coefficients
    a_half = [0.0] * 6  # first 6, then mirror
    b_half = [0.0] * 6
    
    # Use a parameterized ansatz
    # a_half[0] = a_half[5] = edge weight
    # a_half[1] = a_half[4] = inner edge
    # a_half[2] = a_half[3] = center
    
    # Parameterize with p[0], p[1], p[2] for a, p[3], p[4], p[5] for b
    a_edge = p[0]
    a_inner = p[1]
    a_center = p[2]
    
    b_edge = p[3]
    b_inner = p[4]
    b_center = p[5]
    
    a_half = [a_edge, a_inner, a_center, a_center, a_inner, a_edge]
    b_half = [b_edge, b_inner, b_center, b_center, b_inner, b_edge]
    
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Normalize
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    return a, b

def _make_general_symmetric(params):
    """Generate 12-stage coefficients with full mirror symmetry.
    
    params: 12 free parameters (6 for a, 6 for b)
    Returns (a, b) each of length 12
    """
    a_half = list(params[:6])
    b_half = list(params[6:12])
    
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Normalize
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    return a, b

def _nelder_mead(objective, x0, max_iter=200, tol=1e-8):
    """Nelder-Mead optimization with time guard."""
    n = len(x0)
    simplex = [x0[:]]
    step = 0.1
    for i in range(n):
        xi = x0[:]
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    no_improve = 0
    for it in range(max_iter):
        if _remaining() < 10:
            break
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid (excluding worst)
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[n])
        f_ref = objective(x_ref)
        
        if f_ref < fvals[0]:
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp
                fvals[n] = f_exp
            else:
                simplex[n] = x_ref
                fvals[n] = f_ref
            no_improve = 0
        elif f_ref < fvals[n]:
            simplex[n] = x_ref
            fvals[n] = f_ref
            no_improve = 0
        else:
            rho = 0.5
            x_con = centroid + rho * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con
                fvals[n] = f_con
                no_improve = 0
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                no_improve += 1
        
        if no_improve > 30:
            break
    
    # Return best
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error_fast(best_a, best_b)
    
    # --- Phase 1: Try known high-order schemes as starting points ---
    # 4th-order Suzuki-Trotter for 4 stages, repeated 3 times
    c = 6**(1/3)
    alpha1 = 1.0/(6.0 - 6.0*c)
    alpha2 = -1.0/(6.0 - 6.0*c)
    beta1 = 1.0/(2.0 - 6.0*c)
    
    # Pattern for 4 stages
    a_4 = [alpha1, alpha2, alpha2, alpha1]
    b_4 = [beta1, beta1, beta1, beta1]
    
    # Scale to sum to 1
    sa = sum(a_4)
    sb = sum(b_4)
    a_4 = [x/sa for x in a_4]
    b_4 = [x/sb for x in b_4]
    
    # Repeat 3 times to get 12 stages
    a_g = a_4 * 3
    b_g = b_4 * 3
    err = _compute_error_fast(a_g, b_g)
    if err < best_err:
        best_err = err
        best_a = a_g[:]
        best_b = b_g[:]
    
    # 6th-order Yoshida-like (approximate)
    # Use a pattern with more negative weights
    a_6 = [0.3, -0.1, -0.1, -0.1, -0.1, 0.3]
    b_6 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    sa = sum(a_6)
    sb = sum(b_6)
    a_6 = [x/sa for x in a_6]
    b_6 = [x/sb for x in b_6]
    a_g = a_6 * 2
    b_g = b_6 * 2
    err = _compute_error_fast(a_g, b_g)
    if err < best_err:
        best_err = err
        best_a = a_g[:]
        best_b = b_g[:]
    
    # --- Phase 2: Optimize with symmetric parameterization (6 params) ---
    if _remaining() > 30:
        def obj_sym6(params):
            a, b = _make_symmetric_4th_order(params)
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            sa = sum(a)
            sb = sum(b)
            if abs(sa) < 1e-15 or abs(sb) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            b = [x/sb for x in b]
            return _compute_error_fast(a, b)
        
        # Multiple starting points
        starts = [
            [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            [0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
            [0.3, 0.1, 0.1, 0.3, 0.1, 0.1],
            [0.15, 0.25, 0.2, 0.15, 0.25, 0.2],
            [0.25, 0.15, 0.2, 0.25, 0.15, 0.2],
        ]
        
        for start in starts:
            if _remaining() < 15:
                break
            x_opt, f_opt = _nelder_mead(obj_sym6, start, max_iter=100)
            if f_opt < best_err:
                best_err = f_opt
                a_opt, b_opt = _make_symmetric_4th_order(x_opt)
                best_a = a_opt
                best_b = b_opt
    
    # --- Phase 3: Optimize with full symmetric parameterization (12 params) ---
    if _remaining() > 40:
        def obj_sym12(params):
            a, b = _make_general_symmetric(params)
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            sa = sum(a)
            sb = sum(b)
            if abs(sa) < 1e-15 or abs(sb) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            b = [x/sb for x in b]
            return _compute_error_fast(a, b)
        
        # Starting point from best
        # Extract half from best_a and best_b
        a_half = best_a[:6]
        b_half = best_b[:6]
        start = a_half + b_half
        
        x_opt, f_opt = _nelder_mead(obj_sym12, start, max_iter=150)
        if f_opt < best_err:
            best_err = f_opt
            a_opt, b_opt = _make_general_symmetric(x_opt)
            best_a = a_opt
            best_b = b_opt
    
    # --- Phase 4: Fine-tune with full 24 parameters ---
    if _remaining() > 20:
        def obj_full(params):
            a = list(params[:12])
            b = list(params[12:24])
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            sa = sum(a)
            sb = sum(b)
            if abs(sa) < 1e-15 or abs(sb) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            b = [x/sb for x in b]
            return _compute_error_fast(a, b)
        
        start = best_a + best_b
        x_opt, f_opt = _nelder_mead(obj_full, start, max_iter=100)
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # Final normalization and clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
