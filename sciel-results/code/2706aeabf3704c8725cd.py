"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Structural combination:
- Keep the current program's direct 24-parameter (a_0..a_11, b_0..b_11)
  Nelder-Mead machinery with sum-to-1 normalization (the asymmetric
  B-first, A-last ordering is handled by optimizing all 24 independently).
- Enrich the initial-guess family with the inspiration's asymmetric patterns
  and a commutator-derived 4th-order Suzuki-Froehlich seed (correctly scaled).
- ADD a NEW search method: finite-difference gradient-based local descent
  (steepest descent with projected sum constraint) to escape the 24-dim
  local minima that pure simplex methods get stuck in.
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
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    ev_A, vec_A = cache['ev_A'], cache['vec_A']
    ev_B, vec_B = cache['ev_B'], cache['vec_B']
    ev_H, vec_H = cache['ev_H'], cache['vec_H']
    dim = cache['dim']

    times = [0.25, 0.5, 1.0]
    errors = []

    for t in times:
        # Exact propagator
        exact = (vec_H * np.exp(-1j * t * ev_H)) @ vec_H.conj().T

        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            exp_b = (vec_B * np.exp(-1j * b[k] * t * ev_B)) @ vec_B.conj().T
            exp_a = (vec_A * np.exp(-1j * a[k] * t * ev_A)) @ vec_A.conj().T
            U = exp_b @ exp_a @ U

        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)

    return float(np.mean(errors))

def _normalize(x):
    """Split x into (a,b), normalize each to sum 1, clip to [-50,50], renormalize."""
    a = np.array(x[:12], dtype=float)
    b = np.array(x[12:24], dtype=float)
    sa = a.sum()
    sb = b.sum()
    if abs(sa) < 1e-15 or abs(sb) < 1e-15:
        return None
    a = a / sa
    b = b / sb
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    sa = a.sum()
    sb = b.sum()
    if abs(sa) < 1e-15 or abs(sb) < 1e-15:
        return None
    a = a / sa
    b = b / sb
    return np.concatenate([a, b])

def _objective(x):
    n = _normalize(x)
    if n is None:
        return 1e10
    return _compute_error_fast(n[:12].tolist(), n[12:24].tolist())

def _nelder_mead(f, x0, max_iter, step_size, tol=1e-12):
    """Nelder-Mead optimization with time budget check. x0 is a 24-vector."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += step_size
        simplex.append(xi)

    fvals = [f(x) for x in simplex]
    best_val = min(fvals)
    best_x = simplex[int(np.argmin(fvals))][:]
    no_improve = 0

    for it in range(max_iter):
        if _remaining() < 8:
            break

        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if abs(fvals[0] - fvals[-1]) < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)

        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[n])
        f_ref = f(x_ref)

        if f_ref < fvals[0]:
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[n])
            f_exp = f(x_exp)
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
            f_con = f(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con
                fvals[n] = f_con
                no_improve = 0
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                no_improve += 1

        if fvals[0] < best_val:
            best_val = fvals[0]
            best_x = simplex[0][:]

        if no_improve > 40:
            break

    return best_x, best_val

def _gradient_descent(f, x0, max_iter=400, eps=1e-4):
    """Finite-difference steepest descent with sum-constraint projection.

    This is the NEW search method: a gradient-based local refinement that
    complements the simplex method. The sum constraint is enforced by
    projecting each step back onto the normalized manifold via _normalize.
    """
    x = np.array(x0, dtype=float)
    best_x = x[:]
    best_val = f(x)
    lr = 1e-3
    n = len(x)

    for it in range(max_iter):
        if _remaining() < 6:
            break

        # Finite-difference gradient
        grad = np.zeros(n)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (f(xp) - f(xm)) / (2.0 * eps)

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break

        # Step in negative gradient direction
        x_new = x - lr * grad / (gnorm + 1e-12)

        # Project to normalized manifold
        proj = _normalize(x_new)
        if proj is None:
            lr *= 0.5
            continue
        x_new = proj

        f_new = f(x_new)
        if f_new < best_val:
            best_val = f_new
            best_x = x_new[:]
            x = x_new
            # Increase learning rate cautiously if improving
            lr = min(lr * 1.2, 1e-2)
        else:
            lr *= 0.5
            if lr < 1e-7:
                break

    return best_x, best_val

def _suzuki4_seed():
    """Build a 12-stage 4th-order Suzuki-Froehlich seed.

    Standard 4th-order splitting: U(t) = exp(-i beta1 t B) exp(-i alpha1 t A)
        exp(-i beta2 t B) exp(-i alpha2 t A) exp(-i beta1 t B)
    with alpha1 = 1/(2 - 2^(1/3)), alpha2 = -2^(1/3)/(2 - 2^(1/3)),
         beta1 = 1/(2 - 2^(1/3)), beta2 = -2^(1/3)/(2 - 2^(1/3)).
    Here we embed a 4-substep 4th-order pattern (alpha1, alpha2, alpha2, alpha1)
    repeated 3x over 12 stages, normalized to sum 1.
    """
    c = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - c
    alpha1 = 1.0 / denom
    alpha2 = -c / denom
    beta1 = 1.0 / denom
    beta2 = -c / denom

    # 12-stage: 3 copies of a 4-stage 4th-order block
    a = [alpha1, alpha2, alpha2, alpha1] * 3
    b = [beta1, beta2, beta2, beta1] * 3
    sa = sum(a); sb = sum(b)
    if abs(sa) > 1e-12:
        a = [x / sa for x in a]
    if abs(sb) > 1e-12:
        b = [x / sb for x in b]
    return a, b

def coefficients(stages):
    """Optimize 12-stage product formula coefficients.

    Combines:
    1. Multi-start initial guesses (uniform, edge-emphasized, asymmetric,
       and a commutator-derived 4th-order Suzuki-Froehlich seed).
    2. Nelder-Mead (24-dim) from the best seed.
    3. A NEW finite-difference gradient-descent local refinement.
    4. A second, finer Nelder-Mead pass.
    """
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b

    cache = _get_eig()

    # Start with equal weights
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error_fast(best_a, best_b)

    # --- Phase 1: Multi-start with diverse initial guesses ---
    initial_guesses = []

    # Uniform
    initial_guesses.append(([1.0/12]*12, [1.0/12]*12))

    # Edge-emphasized (symmetric)
    for edge_w in [0.5, 1.0, 1.5, 2.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            dist = min(k, 11-k)
            w = 1.0 + 0.3 * (6 - dist) / 6.0 * edge_w
            a_g[k] = 1.0/12 * w
            b_g[k] = 1.0/12 * w
        sa = sum(a_g); sb = sum(b_g)
        a_g = [x/sa for x in a_g]
        b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))

    # Asymmetric patterns (B-first ordering breaks symmetry)
    for asym in [0.2, 0.5, 1.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            a_g[k] = 1.0/12 * (1.0 + asym * k / 11.0)
            b_g[k] = 1.0/12 * (1.0 - asym * k / 11.0)
        sa = sum(a_g); sb = sum(b_g)
        if abs(sa) > 1e-10:
            a_g = [x/sa for x in a_g]
        if abs(sb) > 1e-10:
            b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))

    # Commutator-derived 4th-order Suzuki-Froehlich seed
    s4a, s4b = _suzuki4_seed()
    initial_guesses.append((s4a, s4b))

    # Evaluate each initial guess
    for a_g, b_g in initial_guesses:
        if _remaining() < 5:
            break
        err = _compute_error_fast(a_g, b_g)
        if err < best_err:
            best_err = err
            best_a = a_g[:]
            best_b = b_g[:]

    # --- Phase 2: Nelder-Mead (coarse) ---
    if _remaining() > 30:
        x0 = np.array(best_a + best_b, dtype=float)
        x_nm, f_nm = _nelder_mead(_objective, x0, max_iter=400, step_size=0.02)
        if f_nm < best_err:
            best_err = f_nm
            best_a = x_nm[:12].tolist()
            best_b = x_nm[12:24].tolist()

    # --- Phase 3: NEW gradient-based local refinement ---
    if _remaining() > 20:
        x0 = np.array(best_a + best_b, dtype=float)
        x_gd, f_gd = _gradient_descent(_objective, x0, max_iter=300, eps=1e-4)
        if f_gd < best_err:
            best_err = f_gd
            best_a = x_gd[:12].tolist()
            best_b = x_gd[12:24].tolist()

    # --- Phase 4: Second finer Nelder-Mead ---
    if _remaining() > 15:
        x0 = np.array(best_a + best_b, dtype=float)
        x_nm2, f_nm2 = _nelder_mead(_objective, x0, max_iter=300, step_size=0.005)
        if f_nm2 < best_err:
            best_err = f_nm2
            best_a = x_nm2[:12].tolist()
            best_b = x_nm2[12:24].tolist()

    # --- Final gradient polish if time remains ---
    if _remaining() > 8:
        x0 = np.array(best_a + best_b, dtype=float)
        x_gd2, f_gd2 = _gradient_descent(_objective, x0, max_iter=200, eps=1e-5)
        if f_gd2 < best_err:
            best_err = f_gd2
            best_a = x_gd2[:12].tolist()
            best_b = x_gd2[12:24].tolist()

    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]

    # Clip
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]

    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]

    return best_a, best_b
