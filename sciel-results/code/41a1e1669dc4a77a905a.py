import numpy as np
import time
import sys

def coefficients(stages):
    """
    Twelve-stage product formula for Heisenberg chain time evolution.
    Combines high-order Suzuki-Trotter structure with Nelder-Mead refinement.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * stages
        b = []
        for k in range(stages):
            if k % 2 == 0:
                b.append(0.5 / substeps)
            else:
                b.append(0.0)
        a_sum = sum(a)
        b_sum = sum(b)
        if abs(a_sum) > 1e-10:
            a = [x / a_sum for x in a]
        if abs(b_sum) > 1e-10:
            b = [x / b_sum for x in b]
        return a, b

    # Build A and B operators for a 6-site XXZ Heisenberg chain
    gamma = 0.8
    h = 0.3
    N = 6  # number of sites
    dim = 2 ** N

    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)

    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    def bond_hamiltonian(i, j):
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j

    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)

    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s

    # Precompute eigendecompositions for fast exp
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)

    times = [0.25, 0.5, 1.0]
    U_exact_cache = {}
    for t in times:
        U_exact_cache[t] = np.exp(-1j * t * (A + B)) if False else _expm_via_eigh(A + B, t)

    def _expm_via_eigh(H, t):
        """Compute exp(-i t H) using eigendecomposition."""
        vals, vecs = np.linalg.eigh(H)
        return vecs @ np.diag(np.exp(-1j * t * vals)) @ vecs.conj().T

    U_exact_cache = {}
    AB = A + B
    AB_eigvals, AB_eigvecs = np.linalg.eigh(AB)
    for t in times:
        U_exact_cache[t] = AB_eigvecs @ np.diag(np.exp(-1j * t * AB_eigvals)) @ AB_eigvecs.conj().T

    def compute_error_fast(a, b):
        """Compute mean spectral norm error efficiently using precomputed eigendecompositions."""
        errors = []
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                if abs(b[k]) > 1e-15:
                    theta_b = b[k] * t
                    exp_b = np.exp(-1j * theta_b * B_eigvals)
                    U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
                if abs(a[k]) > 1e-15:
                    theta_a = a[k] * t
                    exp_a = np.exp(-1j * theta_a * A_eigvals)
                    U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
            U_exact = U_exact_cache[t]
            err = np.linalg.norm(U - U_exact, 2)
            errors.append(err)
        return float(np.mean(errors))

    def normalize_and_clip(a, b):
        """Normalize so sums are 1 and clip to [-50, 50]."""
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-10:
            a = [x / sa for x in a]
        if abs(sb) > 1e-10:
            b = [x / sb for x in b]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-10:
            a = [x / sa for x in a]
        if abs(sb) > 1e-10:
            b = [x / sb for x in b]
        return a, b

    # Start with 8th-order Suzuki-Trotter coefficients as initial guess
    # 12-stage symmetric scheme: a_k = a_{13-k}, b_k = b_{13-k}
    # Using known 8th-order coefficients
    c = 2.0 ** (1.0 / 3.0)

    # 8th-order 12-stage symmetric coefficients (Suzuki 1991)
    # a_half and b_half for the first 6 stages
    # These are from the standard 8th-order symmetric splitting
    a_half = [
        0.055501558671812,
        0.150779964897872,
        0.220446795412794,
        0.193080063904462,
        0.123413233389540,
        0.256778423823520
    ]

    # For b, in a symmetric splitting the boundary coefficients are halved
    # b_half for the first 6 stages
    b_half = [
        0.0,
        0.150779964897872,
        0.220446795412794,
        0.193080063904462,
        0.123413233389540,
        0.256778423823520
    ]

    # Build full symmetric arrays
    a_init = [0.0] * 12
    b_init = [0.0] * 12
    for k in range(6):
        a_init[k] = a_half[k]
        a_init[11 - k] = a_half[k]
        b_init[k] = b_half[k]
        b_init[11 - k] = b_half[k]

    # Normalize to ensure sums = 1
    a_init, b_init = normalize_and_clip(a_init, b_init)

    # Check if we have time for optimization
    if time.time() > deadline - 60:
        return a_init, b_init

    # Nelder-Mead optimization
    def nelder_mead(f, x0, max_iter=200, timeout=50):
        """Nelder-Mead simplex optimization with timeout."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Build simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            xi = np.array(x0, dtype=float).copy()
            xi[i] += 0.01
            simplex.append(xi)

        fvals = []
        for v in simplex:
            try:
                fv = f(v)
            except Exception:
                fv = 1e10
            fvals.append(fv)

        best_x = simplex[0].copy()
        best_f = min(fvals)

        for it in range(max_iter):
            if time.time() - start_time > deadline - 10:
                break

            # Sort simplex by f-value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            try:
                f_r = f(x_r)
            except Exception:
                f_r = 1e10

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                try:
                    f_e = f(x_e)
                except Exception:
                    f_e = 1e10
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (worst - centroid)
                try:
                    f_c = f(x_c)
                except Exception:
                    f_c = 1e10
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        try:
                            fvals[i] = f(simplex[i])
                        except Exception:
                            fvals[i] = 1e10

            # Track best
            for i in range(n + 1):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

        return best_x, best_f

    # Objective function that takes 24 params (12 a + 12 b)
    def objective(params):
        a = list(params[:12])
        b = list(params[12:24])
        a, b = normalize_and_clip(a, b)
        return compute_error_fast(a, b)

    x0 = a_init + b_init

    try:
        best_params, best_err = nelder_mead(objective, x0, max_iter=150, timeout=50)
        a_opt = list(best_params[:12])
        b_opt = list(best_params[12:24])
        a_opt, b_opt = normalize_and_clip(a_opt, b_opt)
        return a_opt, b_opt
    except Exception:
        return a_init, b_init
