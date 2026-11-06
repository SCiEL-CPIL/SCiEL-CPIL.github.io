"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    Symmetric 12-stage product formula for a 6-site open XXZ Heisenberg chain
    (anisotropy 0.8, field 0.3) split into odd (A) and even (B) bonds.

    We keep the symmetric 4th-order Suzuki-Trotter family (the current design)
    but refine it: the 12 stages are grouped into a symmetric pattern with a
    small number of free coefficients, and we minimize the mean SQUARED spectral
    norm of the error over the three evaluation times (a smooth surrogate for
    the nonsmooth mean spectral norm) via gradient descent. This corrects the
    earlier flaw of blindly normalizing 4th-order coefficients (which breaks the
    4th-order cancellation) and instead fits the coefficients to the actual
    target times.
    """
    t_start = time.time()
    deadline = t_start + 120.0
    safety = 8.0  # return this many seconds before the 120s wall-clock kill

    # ---- Build the actual Hamiltonian pieces A (odd bonds + half field) and
    # ---- B (even bonds + half field) for a 6-site open XXZ chain.
    N = 6
    aniso = 0.8
    field = 0.3
    dim = 2 ** N

    # Pauli matrices
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        out = np.array([[1.0]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out

    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)

    def bond_ham(i):
        # XXZ bond between site i and i+1: sx_i sx_{i+1} + sy_i sy_{i+1}
        # + aniso * sz_i sz_{i+1}
        ops_x = [I2] * N
        ops_x[i] = sx
        ops_x[i + 1] = sx
        ops_y = [I2] * N
        ops_y[i] = sy
        ops_y[i + 1] = sy
        ops_z = [I2] * N
        ops_z[i] = sz
        ops_z[i + 1] = sz
        return kron_all(ops_x) + kron_all(ops_y) + aniso * kron_all(ops_z)

    # Uniform field term (z-component on every site), split half into A and B.
    field_full = np.zeros((dim, dim), dtype=complex)
    for s in range(N):
        field_full += op_on_site(s, sz)
    half_field = 0.5 * field_full

    # Odd bonds: (0,1), (2,3), (4,5)  -> A
    # Even bonds: (1,2), (3,4)        -> B
    A = half_field.copy()
    for i in (0, 2, 4):
        A += bond_ham(i)
    B = half_field.copy()
    for i in (1, 3):
        B += bond_ham(i)

    H = A + B
    eval_times = [0.25, 0.5, 1.0]

    # Exact propagators (precompute once).
    def exact(t):
        w, V = np.linalg.eigh(H)
        return V @ np.diag(np.exp(-1j * t * w)) @ V.conj().T

    exacts = [exact(t) for t in eval_times]

    # ---- Symmetric 12-stage parameterization.
    # Stages k = 1..12. Symmetric about the center:
    #   a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    #   b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Free params: a1..a6 (6), b1..b6 (6) -> 12 params, then normalize to
    # enforce sum(a)=sum(b)=1 (normalization is part of the smooth map).
    #
    # We parameterize each coefficient as a free real number and apply a
    # normalization; the objective is evaluated on the normalized vectors.

    def build_vectors(p):
        a = p[0:6]
        b = p[6:12]
        a_full = np.array([a[0], a[1], a[2], a[3], a[4], a[5],
                           a[5], a[4], a[3], a[2], a[1], a[0]], dtype=float)
        b_full = np.array([b[0], b[1], b[2], b[3], b[4], b[5],
                           b[5], b[4], b[3], b[2], b[1], b[0]], dtype=float)
        sa = a_full.sum()
        sb = b_full.sum()
        # Guard against zero sums.
        if abs(sa) < 1e-12:
            sa = 1e-12
        if abs(sb) < 1e-12:
            sb = 1e-12
        a_full = a_full / sa
        b_full = b_full / sb
        return a_full, b_full

    def product_formula(t, a_full, b_full):
        # U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = np.exp(-1j * b_full[k] * t * B) @ np.exp(-1j * a_full[k] * t * A) @ U
        return U

    # Precompute the single-step exponentials per stage? They depend on t and
    # coefficients, so compute per call. To speed up, cache exp of A and B at
    # each (coefficient, time) pair via a dict.
    exp_cache = {}

    def mat_exp_cached(M, key):
        if key in exp_cache:
            return exp_cache[key]
        w, V = np.linalg.eigh(M)
        E = V @ np.diag(np.exp(-1j * w)) @ V.conj().T
        exp_cache[key] = E
        return E

    def scaled_exp(base_op, c, t, side):
        # exp(-i * c * t * base_op)
        key = (side, round(c, 12), round(t, 12))
        if key in exp_cache:
            return exp_cache[key]
        M = c * t * base_op
        w, V = np.linalg.eigh(M)
        E = V @ np.diag(np.exp(-1j * w)) @ V.conj().T
        exp_cache[key] = E
        return E

    def product_formula_fast(t, a_full, b_full):
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            Eb = scaled_exp(B, b_full[k], t, 'B')
            Ea = scaled_exp(A, a_full[k], t, 'A')
            U = Eb @ Ea @ U
        return U

    def objective(p):
        a_full, b_full = build_vectors(p)
        # Clip for safety in bounds (objective only; bounds enforced at end).
        err_sq = 0.0
        for t, Ex in zip(eval_times, exacts):
            U = product_formula_fast(t, a_full, b_full)
            E = U - Ex
            # squared spectral norm = largest eigenvalue of E^dagger E
            # Use a cheaper proxy: trace-based is not spectral; use svd max.
            s = np.linalg.svd(E, compute_uv=False)
            err_sq += s[0] ** 2
        return err_sq / len(eval_times)

    def spectral_error(p):
        a_full, b_full = build_vectors(p)
        err = 0.0
        for t, Ex in zip(eval_times, exacts):
            U = product_formula_fast(t, a_full, b_full)
            E = U - Ex
            s = np.linalg.svd(E, compute_uv=False)
            err += s[0]
        return err / len(eval_times)

    # ---- Initial point: standard 4th-order Suzuki-Trotter coefficients
    # ---- mapped into the symmetric 12-stage family (NOT normalized blindly;
    # ---- we let the optimizer handle the fit, but start from a good 4th-order
    # ---- seed so we are in the right basin).
    two_thirds = 2.0 ** (1.0 / 3.0)
    a1_st = 1.0 / (4.0 - two_thirds)
    a2_st = 1.0 / (2.0 * (4.0 - two_thirds))
    b1_st = 1.0 / (2.0 - two_thirds)
    b2_st = 1.0 / (2.0 * (2.0 - two_thirds))

    # Distribute the 4th-order weights into 6 symmetric slots.
    # 12 stages = 2 * 6 unique (mirror). The 4th-order splitting has 2 unique
    # a-values and 2 unique b-values. We spread them so the symmetry holds.
    a_init = [a1_st, a2_st, a2_st, a2_st, a2_st, a2_st]
    b_init = [b1_st, b2_st, b2_st, b2_st, b2_st, b2_st]
    p0 = np.array(a_init + b_init, dtype=float)

    # Normalize the seed so sums are ~1 (helps the optimizer start feasible).
    a_full0, b_full0 = build_vectors(p0)

    # ---- Gradient descent on the smooth squared objective.
    def grad_descent(p, max_iter, tol=1e-10):
        lr = 0.05
        best_p = p.copy()
        best_f = objective(p)
        for it in range(max_iter):
            if time.time() > deadline - safety:
                break
            # Central finite-difference gradient
            g = np.zeros_like(p)
            h = 1e-5
            for i in range(len(p)):
                pp = p.copy(); pp[i] += h
                pm = p.copy(); pm[i] -= h
                g[i] = (objective(pp) - objective(pm)) / (2 * h)
            p = p - lr * g
            # Damping / restart if objective increases
            f = objective(p)
            if not np.isfinite(f) or f > best_f * 1.5:
                p = best_p
                lr *= 0.5
            else:
                if f < best_f:
                    best_f = f
                    best_p = p.copy()
            if abs(f - best_f) < tol:
                break
        return best_p, best_f

    # Run a modest number of iterations (objective is expensive: 3 times x
    # 12 matmuls + SVD each). Keep it bounded by the deadline.
    best_p = p0.copy()
    best_f = objective(p0)

    # A few restarts with small perturbations to escape local minima, all under
    # the deadline guard.
    rng = np.random.default_rng(12345)
    n_restarts = 4
    for r in range(n_restarts):
        if time.time() > deadline - safety:
            break
        if r == 0:
            p_cur = p0.copy()
        else:
            p_cur = p0 + rng.normal(0, 0.05, size=len(p0))
        p_cur, f_cur = grad_descent(p_cur, max_iter=60)
        if f_cur < best_f:
            best_f = f_cur
            best_p = p_cur.copy()

    # ---- Final high-precision polish: a few more gradient steps with a
    # ---- smaller learning rate on the squared objective.
    if time.time() < deadline - safety:
        best_p, _ = grad_descent(best_p, max_iter=40)

    # ---- Map back to the 12-stage vectors and enforce constraints.
    a_full, b_full = build_vectors(best_p)
    a_list = [float(x) for x in a_full]
    b_list = [float(x) for x in b_full]

    # Enforce bounds [-50, 50] and sums = 1.
    a_list = [max(-50.0, min(50.0, x)) for x in a_list]
    b_list = [max(-50.0, min(50.0, x)) for x in b_list]
    sa = sum(a_list)
    sb = sum(b_list)
    if abs(sa) > 1e-12:
        a_list = [x / sa for x in a_list]
    if abs(sb) > 1e-12:
        b_list = [x / sb for x in b_list]

    # Final feasibility check.
    assert len(a_list) == 12 and len(b_list) == 12
    assert abs(sum(a_list) - 1.0) < 1e-8, f"sum(a)={sum(a_list)}"
    assert abs(sum(b_list) - 1.0) < 1e-8, f"sum(b)={sum(b_list)}"
    assert all(-50.0 <= x <= 50.0 for x in a_list)
    assert all(-50.0 <= x <= 50.0 for x in b_list)

    return a_list, b_list
# EVOLVE-BLOCK-END
