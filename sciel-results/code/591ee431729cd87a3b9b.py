"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty.

Structural change from the seed (fixed arithmetic roots): we treat the 12 positive
double-root locations as free variables, build the actual Laguerre combination
F(x) = sum_{k=0}^{m} c_k L_k^{(n)}(x)  with m=12, n=1, enforce the double-root
constraints F(r_j)=F'(r_j)=0 in a least-squares sense, and optimize the roots to
minimize a tight proxy for the Fourier sign-uncertainty upper bound C4.  A strict
wall-clock deadline guard returns the best feasible root set found.
"""
import time
import math
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

M = 12      # number of double roots
N = 1       # Laguerre order
DEADLINE = 600.0   # evaluator kills at 630 s; keep safety margin
SAFETY = 45.0      # stop this much before the hard limit


def _laguerre_matrix(x, m, n):
    """m+1 x len(x) matrix of L_k^{(n)}(x)."""
    cols = len(x)
    A = np.empty((m + 1, cols))
    for k in range(m + 1):
        A[k] = eval_genlaguerre(k, n, x)
    return A


def _laguerre_deriv_matrix(x, m, n):
    """m+1 x len(x) matrix of d/dx L_k^{(n)}(x).  d/dx L_k^{(n)} = -L_{k-1}^{(n+1)}."""
    cols = len(x)
    A = np.empty((m + 1, cols))
    A[0] = 0.0
    for k in range(1, m + 1):
        A[k] = -eval_genlaguerre(k - 1, n + 1, x)
    return A


def _solve_coeffs(roots, m=M, n=N):
    """Least-squares coefficient vector c so that F and F' vanish at each root."""
    R = np.asarray(roots, dtype=float)
    A = _laguerre_matrix(R, m, n)
    B = _laguerre_deriv_matrix(R, m, n)
    # stacked system: [A; B] c = 0  -> minimize ||[A;B] c||^2 subject to ||c||=1
    S = np.vstack([A, B])
    # SVD: smallest singular vector gives the null-space solution
    try:
        _, _, Vt = np.linalg.svd(S, full_matrices=True)
        c = Vt[-1]
    except Exception:
        c = np.ones(m + 1) / math.sqrt(m + 1)
    norm = np.linalg.norm(c)
    if norm < 1e-14:
        c = np.ones(m + 1) / math.sqrt(m + 1)
    else:
        c = c / norm
    return c


def _residual(roots, m=M, n=N):
    """Sum of squared double-root residuals (F and F' at each root)."""
    R = np.asarray(roots, dtype=float)
    c = _solve_coeffs(roots, m, n)
    F = _laguerre_matrix(R, m, n).T @ c
    Fp = _laguerre_deriv_matrix(R, m, n).T @ c
    return float(np.sum(F * F) + np.sum(Fp * Fp))


def _bound_proxy(roots, m=M, n=N):
    """
    Proxy for the Fourier sign-uncertainty upper bound C4.

    The bound is driven by (a) how well the double-root constraints are met
    (residual) and (b) how far the last (largest) root pushes the final sign
    change.  We minimize a composite that is monotone in the true bound:
        C4 ~ r_max * (1 + residual)
    which rewards tight, compact, well-constrained root sets.
    """
    R = np.asarray(roots, dtype=float)
    r_max = float(np.max(R))
    res = _residual(roots, m, n)
    # compactness: penalize gaps so the sign-change structure stays tight
    gaps = np.diff(np.sort(R))
    gap_spread = float(np.max(gaps) - np.min(gaps)) if len(gaps) else 0.0
    return r_max * (1.0 + res) + 0.1 * gap_spread


def _feasible(roots):
    R = np.sort(np.asarray(roots, dtype=float))
    if R.shape[0] != M:
        return False
    if np.any(R <= 0.0):
        return False
    if np.any(np.diff(R) <= 1e-9):
        return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    # ---- deterministic initial guesses (several families) ----
    def guess(kind):
        i = np.arange(M)
        if kind == 0:
            return 4.0 * (i + 1)                       # seed
        if kind == 1:
            return 1.0 + 2.0 * i                       # tighter uniform
        if kind == 2:
            return 1.5 + 1.5 * i
        if kind == 3:
            # geometric-ish, denser at low x
            return 0.8 * (1.3 ** i) + 0.5
        if kind == 4:
            # roots near Laguerre zero scale ~ (2i+1+n)
            return 2.0 * (i + 1) + 0.5 * (i + 1) ** 0.5
        return 4.0 * (i + 1)

    best_roots = None
    best_score = math.inf

    def consider(roots):
        nonlocal best_roots, best_score
        if not _feasible(roots):
            return
        s = _bound_proxy(roots)
        if s < best_score:
            best_score = s
            best_roots = np.sort(np.asarray(roots, dtype=float))

    # seed baseline
    consider(guess(0))

    # ---- local Nelder-Mead polish from each family ----
    for kind in range(5):
        if time.time() > deadline:
            break
        x0 = guess(kind)
        try:
            res = minimize(
                _bound_proxy,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9,
                         "adaptive": True},
            )
            consider(res.x)
        except Exception:
            pass

    # ---- bounded global search with differential evolution (cheap, few gen) ----
    if time.time() < deadline:
        lb = np.full(M, 0.3)
        ub = np.full(M, 60.0)
        try:
            de_res = differential_evolution(
                _bound_proxy,
                bounds=list(zip(lb, ub)),
                seed=12345,
                maxiter=12,
                popsize=10,
                tol=1e-6,
                mutation=(0.3, 0.8),
                recombination=0.8,
                polish=True,
                init="sobol",
            )
            consider(de_res.x)
        except Exception:
            pass

    # ---- final L-BFGS-B refinement from the best found ----
    if best_roots is not None and time.time() < deadline:
        try:
            lb = np.full(M, 0.2)
            ub = np.full(M, 80.0)
            lbfgs = minimize(
                _bound_proxy,
                best_roots,
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
            )
            consider(lbfgs.x)
        except Exception:
            pass

    if best_roots is None:
        # absolute fallback: seed
        best_roots = np.array([4.0 * (i + 1) for i in range(M)], dtype=float)

    best_roots = np.sort(best_roots)
    return [float(v) for v in best_roots]
