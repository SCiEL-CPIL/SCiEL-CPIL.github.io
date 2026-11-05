"""Product-formula design for Heisenberg-chain time evolution.

Uses eigen-decomposition of the Hermitian halves A and B for fast
exponential evaluation, then optimizes 12-stage coefficients via
Nelder-Mead with palindromic symmetry and sum constraints.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    n = 6
    dim = 2 ** n
    d = 0.8
    h = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)

    site_ops = {}
    for site in range(n):
        for name, op in (("sx", sx), ("sy", sy), ("sz", sz)):
            ops = [ident] * n
            ops[site] = op
            r = ops[0]
            for o in ops[1:]:
                r = np.kron(r, o)
            site_ops[(site, name)] = r

    def bond_op(i, j):
        return (site_ops[(i, "sx")] @ site_ops[(j, "sx")]
                + site_ops[(i, "sy")] @ site_ops[(j, "sy")]
                + d * site_ops[(i, "sz")] @ site_ops[(j, "sz")])

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)

    half_field = 0.5 * h
    for site in range(n):
        A += half_field * site_ops[(site, "sz")]
        B += half_field * site_ops[(site, "sz")]

    return A, B


def _eig_decomp(M):
    """Eigendecomposition of a Hermitian matrix M: M = V diag(w) V^H."""
    w, V = np.linalg.eigh(M)
    return w, V


def _expm_hermitian(w, V, coeff):
    """Compute exp(coeff * M) where M = V diag(w) V^H."""
    if abs(coeff) < 1e-15:
        n = len(w)
        return np.eye(n, dtype=complex)
    e = np.exp(coeff * w)
    return (V * e) @ V.conj().T


def _symmetrize(params, n):
    """Expand half-parameters to full palindromic a, b lists."""
    half = n // 2
    a = np.zeros(n)
    b = np.zeros(n)
    for k in range(half):
        a[k] = params[k]
        a[n - 1 - k] = params[k]
        b[k] = params[half + k]
        b[n - 1 - k] = params[half + k]
    return a, b


def _compute_error(a, b, A_eig, B_eig, Uexact, times):
    """Compute mean spectral-norm error over given times."""
    n = A_eig[1].shape[0]
    I = np.eye(n, dtype=complex)
    errors = []
    for t in times:
        U = I.copy()
        for k in range(len(a)):
            U = _expm_hermitian(B_eig[0], B_eig[1], -1j * b[k] * t) @ \
                _expm_hermitian(A_eig[0], A_eig[1], -1j * a[k] * t) @ U
        errors.append(np.linalg.norm(U - Uexact[t], 2))
    return float(np.mean(errors))


def _nelder_mead(objective, x0, max_iter=100, tol=1e-8, deadline=None):
    """Standard Nelder-Mead simplex method."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        x_new[i] += step
        simplex.append(x_new)
    fvals = [objective(x) for x in simplex]

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    return simplex[0], fvals[0]


def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150 s wall-clock

    n = stages
    A, B = _build_hamiltonians()
    H = A + B

    # Eigen-decompose A and B (Hermitian) for fast exponentials
    A_eig = _eig_decomp(A)
    B_eig = _eig_decomp(B)
    H_eig = _eig_decomp(H)

    times = [0.25, 0.5, 1.0]
    Uexact = {t: _expm_hermitian(H_eig[0], H_eig[1], -1j * t) for t in times}

    half = n // 2
    # Initial guess: uniform
    x0 = np.array([1.0 / n] * half + [1.0 / n] * half)

    def objective(params):
        a, b = _symmetrize(params, n)
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        sa, sb = a.sum(), b.sum()
        if abs(sa - 1.0) > 1e-8 or abs(sb - 1.0) > 1e-8:
            return 1e10
        return _compute_error(a, b, A_eig, B_eig, Uexact, times)

    best_a, best_b = _symmetrize(x0, n)
    best_err = objective(x0)

    # Nelder-Mead with multiple restarts
    rng = np.random.default_rng(42)
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        if r == 0:
            x_cur = x0.copy()
        elif r == 1:
            # Slightly non-uniform perturbation
            x_cur = x0 + rng.normal(0, 0.02, size=len(x0))
        elif r == 2:
            x_cur = x0 + rng.normal(0, 0.05, size=len(x0))
        elif r == 3:
            x_cur = x0 + rng.normal(0, 0.1, size=len(x0))
        else:
            x_cur = x0 + rng.normal(0, 0.03, size=len(x0))

        try:
            x_opt, f_opt = _nelder_mead(objective, x_cur, max_iter=80,
                                        tol=1e-8, deadline=deadline - 20.0)
            if f_opt < best_err:
                best_err = f_opt
                best_a, best_b = _symmetrize(x_opt, n)
        except Exception:
            pass

    # Final constraint cleanup: renormalize and clamp
    for arr_name in ("a", "b"):
        arr = best_a if arr_name == "a" else best_b
        arr = np.clip(arr, -50.0, 50.0)
        s = arr.sum()
        if abs(s) > 1e-12:
            arr = arr / s
        else:
            arr = np.ones(n) / n
        if arr_name == "a":
            best_a = arr
        else:
            best_b = arr

    return best_a.tolist(), best_b.tolist()
# EVOLVE-BLOCK-END
