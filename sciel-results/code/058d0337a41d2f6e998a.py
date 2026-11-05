"""Optimized 12-stage product formula for 6-site XXZ Heisenberg chain.
Uses eigendecomposition-based matrix exponential for speed, and a
constrained optimization with precomputed spectral decompositions."""

import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field).
    6-site open XXZ chain, anisotropy gamma=0.8, field h=0.3."""
    N = 6
    gamma = 0.8
    h = 0.3
    dim = 2 ** N

    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def op_on_site(op, site):
        result = np.array([[1]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, I2)
        return result

    def bond_op(i, j):
        Sxi = op_on_site(Sx, i)
        Sxj = op_on_site(Sx, j)
        Syi = op_on_site(Sy, i)
        Syj = op_on_site(Sy, j)
        Szi = op_on_site(Sz, i)
        Szj = op_on_site(Sz, j)
        return (gamma/2)*(Sxi @ Sxj + Syi @ Syj) + 0.5*(Szi @ Szj) + h/2*(Szi + Szj)

    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    B = bond_op(1, 2) + bond_op(3, 4)

    # Add half field to each
    half_field = h / 2
    for site in range(N):
        Szi = op_on_site(Sz, site)
        A += half_field * Szi
        B += half_field * Szi

    return A, B

def _eigdecomp(M):
    """Compute eigenvalue decomposition for Hermitian M: M = V diag(w) V^H."""
    w, V = np.linalg.eigh(M)
    return w, V

def _expm_from_eig(c, w, V):
    """Compute exp(-i*c*M) given eigendecomposition M = V diag(w) V^H."""
    # exp(-i*c*M) = V diag(exp(-i*c*w)) V^H
    phases = np.exp(-1j * c * w)
    # V @ diag(phases) @ V^H
    return (V * phases) @ V.conj().T

def _build_hamiltonians_with_eig():
    """Build A, B and their eigendecompositions, plus exact propagator."""
    A, B = _build_hamiltonians()
    H = A + B
    wA, VA = _eigdecomp(A)
    wB, VB = _eigdecomp(B)
    wH, VH = _eigdecomp(H)
    times = [0.25, 0.5, 1.0]
    Uexact = {t: _expm_from_eig(t, wH, VH) for t in times}
    return A, B, wA, VA, wB, VB, times, Uexact

def _compute_error_fast(a, b, wA, VA, wB, VB, times, Uexact):
    """Compute mean spectral-norm error using precomputed eigendecompositions."""
    dim = len(wA)
    I = np.eye(dim, dtype=complex)
    
    total_err = 0.0
    for t in times:
        U = I.copy()
        for k in range(12):
            if abs(b[k]) > 1e-15:
                U = _expm_from_eig(b[k] * t, wB, VB) @ U
            if abs(a[k]) > 1e-15:
                U = _expm_from_eig(a[k] * t, wA, VA) @ U
        err = np.linalg.norm(U - Uexact[t], 2)
        total_err += err
    return total_err / len(times)

def _symmetrize(params, n):
    """Expand half-parameters to full palindromic a, b lists of length n."""
    half = n // 2
    a = np.zeros(n)
    b = np.zeros(n)
    for k in range(half):
        a[k] = params[k]
        a[n - 1 - k] = params[k]
        b[k] = params[half + k]
        b[n - 1 - k] = params[half + k]
    return a, b

def _nelder_mead(objective, x0, max_iter=100, tol=1e-8, deadline=None):
    """Nelder-Mead simplex optimization."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 if abs(x0[i]) < 1e-10 else 0.05 * max(abs(x0[i]), 0.1)
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
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    return simplex[0], fvals[0]

def coefficients(stages):
    """Return optimized (a, b) lists of length `stages` (12)."""
    global _START, _DEADLINE

    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    A, B, wA, VA, wB, VB, times, Uexact = _build_hamiltonians_with_eig()

    n = 12
    half = n // 2

    # Initial guess: uniform Strang-like with palindromic symmetry
    x0 = np.array([1.0 / n] * half + [1.0 / n] * half)

    def objective(params):
        a, b = _symmetrize(params, n)
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        sa, sb = a.sum(), b.sum()
        if abs(sa - 1.0) > 1e-6 or abs(sb - 1.0) > 1e-6:
            return 1e10
        return _compute_error_fast(a, b, wA, VA, wB, VB, times, Uexact)

    best_a, best_b = _symmetrize(x0, n)
    best_err = objective(x0)

    # Nelder-Mead with multiple restarts
    rng = np.random.default_rng(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 5.0:
            break
        if r == 0:
            x_cur = x0.copy()
        else:
            # Perturb with decreasing magnitude
            sigma = 0.1 / (r + 1)
            x_cur = x0 + rng.normal(0, sigma, size=len(x0))
        try:
            x_opt, f_opt = _nelder_mead(objective, x_cur, max_iter=80,
                                        tol=1e-8, deadline=_DEADLINE - 5.0)
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
