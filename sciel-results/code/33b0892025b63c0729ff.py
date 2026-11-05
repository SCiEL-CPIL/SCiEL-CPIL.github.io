"""Combined program: build exact Hamiltonians, precompute spectral data, then
optimize 12-stage symmetric product-formula coefficients with Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_N = 6
_DIM = 2 ** _N
_LAM = 0.8
_H_FIELD = 0.3
_TIMES = (0.25, 0.5, 1.0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    N = _N
    dim = _DIM
    lam = _LAM
    h = _H_FIELD

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    I2 = np.eye(2, dtype=complex)

    def two_site(op1, s1, op2, s2):
        ops = [I2] * N
        ops[s1] = op1
        ops[s2] = op2
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond(s1, s2):
        return (two_site(Sx, s1, Sx, s2) +
                two_site(Sy, s1, Sy, s2) +
                lam * two_site(Sz, s1, Sz, s2))

    def one_site(op, s):
        ops = [I2] * N
        ops[s] = op
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Odd bonds: (0,1), (2,3), (4,5) -> A
    A += bond(0, 1)
    A += bond(2, 3)
    A += bond(4, 5)

    # Even bonds: (1,2), (3,4) -> B
    B += bond(1, 2)
    B += bond(3, 4)

    # Field: half to each
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += one_site(Sz, j)

    A += 0.5 * h * H_field
    B += 0.5 * h * H_field

    return A, B


def _precompute():
    """Eigen-decompose A, B, and A+B; return data needed for fast evaluation."""
    A, B = _build_hamiltonians()
    H = A + B

    # Eigen-decompose A and B
    a_vals, a_vecs = np.linalg.eigh(A)
    b_vals, b_vecs = np.linalg.eigh(B)
    h_vals, h_vecs = np.linalg.eigh(H)

    # Precompute exact propagators
    U_exact = []
    for t in _TIMES:
        phase = np.exp(-1j * t * h_vals)
        U_t = (h_vecs * phase) @ h_vecs.conj().T
        U_exact.append(U_t)

    # Precompute exp(-i t A) and exp(-i t B) factor matrices for each t
    # For each time t and each coefficient value, we need exp(-i c*t*A)
    # We'll compute on the fly using eigen-decomp: V diag(exp(-i c*t*vals)) V^H
    # But to speed up, precompute the "phase vectors" for each t
    # Actually, for the objective we need exp(-i a_k * t * A) for various a_k
    # Let's precompute V_A, vals_A, V_B, vals_B so we can do:
    # exp(-i c*t*A) = (V_A * exp(-i c*t*vals_A)) @ V_A^H

    return {
        'a_vals': a_vals, 'a_vecs': a_vecs,
        'b_vals': b_vals, 'b_vecs': b_vecs,
        'h_vals': h_vals, 'h_vecs': h_vecs,
        'U_exact': U_exact,
        'A': A, 'B': B, 'H': H,
    }


def _exp_op(vals, vecs, c, t):
    """Compute exp(-i c t H) given eigenvalues vals and eigenvectors vecs."""
    phase = np.exp(-1j * c * t * vals)
    return (vecs * phase) @ vecs.conj().T


def _objective(params, data):
    """
    Compute mean spectral-norm error over the three times.
    params: 22 floats (11 free a + 11 free b), last a and b determined by sum=1.
    """
    a_free = params[:11]
    b_free = params[11:22]

    a = list(a_free)
    a.append(1.0 - np.sum(a_free))
    b = list(b_free)
    b.append(1.0 - np.sum(b_free))

    # Bounds check
    if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
        return 1e10

    a_vals = data['a_vals']
    a_vecs = data['a_vecs']
    b_vals = data['b_vals']
    b_vecs = data['b_vecs']
    U_exact = data['U_exact']

    err_sum = 0.0
    for idx, t in enumerate(_TIMES):
        U_approx = np.eye(_DIM, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            phase_b = np.exp(-1j * b[k] * t * b_vals)
            expB = (b_vecs * phase_b) @ b_vecs.conj().T
            U_approx = expB @ U_approx
            # exp(-i a_k t A)
            phase_a = np.exp(-1j * a[k] * t * a_vals)
            expA = (a_vecs * phase_a) @ a_vecs.conj().T
            U_approx = expA @ U_approx

        diff = U_approx - U_exact[idx]
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        err_sum += s[0]

    return err_sum / len(_TIMES)


def _nelder_mead(f, x0, max_iter=300, alpha=1.0, gamma=2.0, rho=0.5,
                 sigma=0.5, tol=1e-10, deadline=None):
    """Nelder-Mead optimization with time guard."""
    n = len(x0)

    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05
        x_new[i] += step
        simplex.append(x_new)

    values = [f(x) for x in simplex]

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Sort
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if abs(values[0] - values[-1]) < tol:
            break

        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]

        x_centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)

        if f_r < values[0]:
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])

    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]


def coefficients(stages):
    """
    Return coefficients (a, b) for the 12-stage product formula.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    if stages != 12:
        # Fallback: uniform
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    data = _precompute()

    # Initial guess: symmetric pattern based on Yoshida 4th-order
    # x = 1/(2 - 2^(1/3))
    x = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    # Base 4th-order: a = [x/2, (1-x)/2, x/2], b = [x/2, (1-x)/2, x/2]
    # Map to 12-stage symmetric: a = [a1,a2,a3,a4,a5,a6,a6,a5,a4,a3,a2,a1]
    # Use a simple symmetric initialization
    # Start with equal weights, then optimize
    a_init = [1.0 / 12.0] * 12
    b_init = [1.0 / 12.0] * 12

    # Free params: first 11 of each, last determined
    x0 = np.array(a_init[:11] + b_init[:11])

    def obj(p):
        return _objective(p, data)

    # Run Nelder-Mead
    best_x, best_val = _nelder_mead(obj, x0, max_iter=500, deadline=deadline)

    # Extract a, b
    a = list(best_x[:11])
    a.append(1.0 - np.sum(a))
    b = list(best_x[11:22])
    b.append(1.0 - np.sum(b))

    # Clip to bounds (safety)
    a = [max(-50.0, min(50.0, ai)) for ai in a]
    b = [max(-50.0, min(50.0, bi)) for bi in b]

    # Re-normalize to ensure sum=1 exactly
    a_sum = sum(a)
    b_sum = sum(b)
    if abs(a_sum) > 1e-15:
        a = [ai / a_sum for ai in a]
    if abs(b_sum) > 1e-15:
        b = [bi / b_sum for bi in b]

    return a, b
# EVOLVE-BLOCK-END
