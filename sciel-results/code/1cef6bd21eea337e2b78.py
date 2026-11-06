import numpy as np
import time

def coefficients(stages):
    if stages != 12:
        return [1.0/stages]*stages, [1.0/stages]*stages

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin from 60s budget

    # Construct A and B for the 6-site Heisenberg chain
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3

    # Spin operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def kron_chain(ops):
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Field part: half in A, half in B
    field_term = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        ops = [identity]*N
        ops[i] = sz
        field_term += kron_chain(ops)

    half_field = 0.5 * field * field_term

    # Bonds
    def add_bond(mat, i, j):
        # Sx_i Sx_j + Sy_i Sy_j + anisotropy Sz_i Sz_j
        ops_xx = [identity]*N
        ops_xx[i] = sx
        ops_xx[j] = sx
        mat += kron_chain(ops_xx)

        ops_yy = [identity]*N
        ops_yy[i] = sy
        ops_yy[j] = sy
        mat += kron_chain(ops_yy)

        ops_zz = [identity]*N
        ops_zz[i] = sz
        ops_zz[j] = sz
        mat += anisotropy * kron_chain(ops_zz)

    # A: Odd bonds (0,1), (2,3), (4,5)
    A += half_field
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)

    # B: Even bonds (1,2), (3,4)
    B += half_field
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)

    # Exact propagator
    H = A + B

    # Precompute exp for various coefficients to speed up evaluation
    # We'll cache matrix exponentials for a_k and b_k values

    def calc_error(a, b):
        """Calculate mean spectral norm error at t = 0.25, 0.5, 1.0"""
        err = 0.0
        for t in [0.25, 0.5, 1.0]:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = U @ np.linalg.matrix_exp(-1j * b[k] * t * B) @ np.linalg.matrix_exp(-1j * a[k] * t * A)
            U_exact = np.linalg.matrix_exp(-1j * t * H)
            # Spectral norm via SVD
            s = np.linalg.svd(U - U_exact, compute_uv=False)
            err += s[0]
        return err / 3.0

    # Try scipy optimization first
    try:
        import scipy.optimize as opt
        use_scipy = True
    except ImportError:
        use_scipy = False

    if use_scipy and time.time() < deadline:
        # Initial guess: 4th-order Yoshida coefficients
        # For 12-stage symmetric 4th-order scheme
        alpha = 2.0 ** (1.0/3.0)
        c1 = 1.0 / (2.0 * (1.0 - alpha))
        c2 = -alpha / (2.0 * (1.0 - alpha))
        c3 = 1.0 / (2.0 * (1.0 + alpha))

        # The 12-stage 4th-order Yoshida coefficients (symmetric)
        # Pattern: a1, a2, a3, a3, a2, a1, a1, a2, a3, a3, a2, a1
        # But we need to normalize so sum = 1
        raw_a = [c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1]
        raw_sum = sum(raw_a)
        a0 = [x / raw_sum for x in raw_a]
        b0 = a0[:]  # Same coefficients for B

        x0 = [a0[0], a0[1], a0[2], a0[3], a0[4], a0[5],
              b0[0], b0[1], b0[2], b0[3], b0[4], b0[5]]
        # Parameterize: a12 = 1 - sum(a1..a11), b12 = 1 - sum(b1..b11)
        # So x = [a1..a11, b1..b11]
        x0 = a0[:11] + b0[:11]

        def cost_func(x):
            if time.time() > deadline:
                return 1e6
            a = list(x[:11])
            b = list(x[11:])
            a.append(1.0 - sum(a))
            b.append(1.0 - sum(b))
            # Check bounds
            for val in a + b:
                if abs(val) > 50 or not np.isfinite(val):
                    return 1e6
            return calc_error(a, b)

        bounds = [(-50.0, 50.0)] * 22

        best_x = x0[:]
        best_cost = cost_func(x0)

        # Use Nelder-Mead for robustness
        res = opt.minimize(cost_func, x0, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})

        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x

        a = list(best_x[:11])
        b = list(best_x[11:])
        a.append(1.0 - sum(a))
        b.append(1.0 - sum(b))

        # Verify validity
        if all(np.isfinite(a)) and all(np.isfinite(b)) and all(abs(x) <= 50 for x in a + b):
            # Check sums
            if abs(sum(a) - 1.0) < 1e-10 and abs(sum(b) - 1.0) < 1e-10:
                return a, b

    # Fallback: 4th-order Yoshida coefficients
    alpha = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = -alpha / (2.0 * (1.0 - alpha))
    c3 = 1.0 / (2.0 * (1.0 + alpha))

    raw_a = [c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1]
    raw_sum = sum(raw_a)
    a = [x / raw_sum for x in raw_a]
    b = a[:]

    return a, b
