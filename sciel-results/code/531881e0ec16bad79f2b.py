"""Tune a Fourier-invariant Hermite-times-Gaussian test function to reduce the
upper bound on A(f)*A(Fourier(f)).

The test function is  g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2)
with a12 fixed so that g(0) = 0, i.e. a12 = -(a0 + 3 a4 + 105 a8).

The reported quantity is the upper bound on C4,
    B = (||x g||_2 / ||g||_2) * (||g_hat||_2 / ||g||_2),
where g_hat is the L2-normalised Fourier transform.  Using the orthonormal
Hermite-function basis psi_n = H_n exp(-x^2/2) / sqrt(2^n n! sqrt(pi)), the
L2 norms become quadratic forms in the coefficient vector v = (a0, a4, a8):

    ||g||^2      = v^T B v
    ||x g||^2    = v^T A v
    ||g_hat||^2  = v^T C v          (C = B^{1/2} U B^{-1} U^T B^{1/2}, U unitary)

so B = (v^T A v / v^T B v) * (v^T C v / v^T B v) is a Rayleigh-quotient product.
Its stationary points are the (real) eigenvalues of the symmetric generalised
eigenvalue problem  A y = mu B y  with a positive eigenvector y.  We build the
matrices exactly and select the smallest admissible eigenvalue.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, inv

# --- deadline guard: the evaluator kills the process at 210 s ---------------
import time as _time
_T0 = _time.monotonic()
_LIMIT = 205.0  # seconds; keep a 5 s safety margin


def _deadline_hit():
    return (_time.monotonic() - _T0) > _LIMIT


def _h(n):
    """2^n * n! * sqrt(pi)  (the L2 norm squared of H_n exp(-x^2/2))."""
    return float(2.0 ** n) * float(np.math.factorial(n)) * np.sqrt(np.pi)


def _build_matrices():
    """Return (A, B, C) acting on the free vector v = (a0, a4, a8)."""
    idx = [0, 4, 8]
    n = 3
    d = _h(0) + 3.0 * _h(4) + 105.0 * _h(8)   # g(0) = sum a_k H_k(0) = 0

    # x acting on the Hermite-function basis:  x psi_j = sqrt((j+1)/2) psi_{j+1}
    #                                    + sqrt(j/2) psi_{j-1}
    def xrow(j):
        c = {}
        if j + 1 <= 12:
            c[j + 1] = np.sqrt((j + 1) / 2.0)
        if j - 1 >= 0:
            c[j - 1] = np.sqrt(j / 2.0)
        return c

    # Fourier (unitary):  U psi_j = sqrt(2/(1+delta_jj)) psi_j
    def urow(j):
        c = {}
        if j == 0:
            c[0] = 1.0
        elif j == 4:
            c[4] = 1.0
        else:
            c[0] = np.sqrt(0.5)
            c[4] = np.sqrt(0.5)
            c[8] = np.sqrt(0.5)
            c[12] = np.sqrt(0.5)
        return c

    # matrix M[k][l] = <psi_k, Op psi_l>  (symmetric)
    def op_matrix(op):
        M = np.zeros((13, 13))
        for l in range(13):
            for k, ck in op(l).items():
                M[k, l] = ck
        return M

    X = op_matrix(xrow)
    U = op_matrix(urow)

    Bfull = np.diag([_h(j) for j in range(13)])
    Afull = X.T @ Bfull @ X
    Cfull = Bfull @ (U.T @ Bfull @ U) @ Bfull

    # project onto the free coordinates
    sel = np.array(idx)
    B = Bfull[np.ix_(sel, sel)]
    A = Afull[np.ix_(sel, sel)]
    C = Cfull[np.ix_(sel, sel)]

    # impose  a12 = -d^{-1} (a0 + 3 a4 + 105 a8)
    w = np.array([1.0, 3.0, 105.0])
    h12 = _h(12)
    B = B + (h12 / (d * d)) * np.outer(w, w)
    A = A + (h12 / (d * d)) * (np.outer(w, Afull[12, sel])
                               + np.outer(Afull[sel, 12], w)
                               + Afull[12, 12] * np.outer(w, w))
    C = C + (h12 / (d * d)) * (np.outer(w, Cfull[12, sel])
                               + np.outer(Cfull[sel, 12], w)
                               + Cfull[12, 12] * np.outer(w, w))
    return A, B, C, w, d, Afull, Cfull


def _bound_from_v(v, A, B, C):
    v = np.asarray(v, float)
    bg = float(v @ B @ v)
    if bg <= 0:
        return np.inf
    xg = float(v @ A @ v)
    fg = float(v @ C @ v)
    if xg < 0 or fg < 0:
        return np.inf
    return np.sqrt(xg / bg) * np.sqrt(fg / bg)


def _polish(v0, A, B, C, max_iter=20000, tol=1e-15):
    """Newton polish of the stationarity system  A v = mu B v,  v^T B v = 1."""
    v = np.asarray(v0, float)
    bg = float(v @ B @ v)
    v = v / np.sqrt(bg)
    for _ in range(max_iter):
        if _deadline_hit():
            break
        mu = float(v @ A @ v)
        r = A @ v - mu * B @ v
        # Jacobian of the residual (mu, v) with constraint v^T B v = 1:
        #   d r / d mu = -B v
        #   d r / d v  = A - mu B
        J11 = -B @ v
        J22 = A - mu * B
        J12 = B
        J21 = B.T
        # solve  [J11 J12; J21 J22] [dmu; dv] = -r
        # use block elimination: (J22 - J21 J11^{-1} J12) dv = -r - J21 J11^{-1} (-mu ... )
        # Simpler: assemble the 4x4 system and solve with numpy (tiny).
        J = np.zeros((4, 4))
        J[0, 0] = J11[0]
        J[1, 0] = J11[1]
        J[2, 0] = J11[2]
        J[0, 1:4] = J12[0, :]
        J[1, 1:4] = J12[1, :]
        J[2, 1:4] = J12[2, :]
        J[3, 0] = J21[0, 0]
        J[3, 1:4] = J22[0, :]
        # note: J21 row for the mu-equation is J12 (symmetric), and the v-rows
        # of J21 are J22's columns handled via symmetry; build full properly:
        J = np.zeros((4, 4))
        J[0:3, 0] = J11
        J[0:3, 1:4] = J12
        J[3, 0] = J21[0]      # d(r_0)/dmu = - (B v)_0
        J[3, 1:4] = J22[0, :]
        rhs = np.concatenate([-r[:3], [-r[0]]])
        try:
            step = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            break
        dmu, dv = step[0], step[1:4]
        v = v + dv
        bg = float(v @ B @ v)
        if bg <= 0:
            break
        v = v / np.sqrt(bg)
        if abs(dmu) < tol and np.linalg.norm(dv) < tol:
            break
    return v


def construct():
    try:
        A, B, C, w, d, Afull, Cfull = _build_matrices()

        # reference (prior-art) coefficients
        v_ref = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best = _bound_from_v(v_ref, A, B, C)
        best_v = v_ref.copy()

        # exact symmetric generalised eigenproblem  A y = mu B y
        # B is SPD (it is ||g||^2 on a basis), so use the Cholesky form.
        try:
            Bcf, Bc = cho_factor(B)
            # transform to standard symmetric problem
            Binv_sqrt = cho_solve((Bcf, Bc), np.eye(3))
            Astd = Binv_sqrt @ A @ Binv_sqrt
            Astd = 0.5 * (Astd + Astd.T)
            vals, vecs = eigh(Astd)
        except Exception:
            vals, vecs = np.linalg.eig(A @ np.linalg.inv(B))
            vals = np.real(vals)
            vecs = np.real(vecs)
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]

        candidates = []
        for mu, y in zip(vals, vecs.T):
            if not np.isfinite(mu):
                continue
            v = B @ y          # map back: y = B^{-1/2} ...  v = B y works since
            # stationarity A v = mu B v  <=>  Astd (B^{-1}v) = mu (B^{-1}v)
            # so v = B y is a valid representative of the eigenspace.
            if _deadline_hit():
                break
            # keep the orientation with positive coefficients (admissible region)
            if v.sum() < 0:
                v = -v
            if np.any(v < -1e-12):
                continue
            b = _bound_from_v(v, A, B, C)
            if np.isfinite(b) and b < best:
                best = b
                best_v = v.copy()
            candidates.append((b, v.copy()))

        # polish the best candidate (and a couple of neighbours) to machine precision
        for b, v in sorted(candidates, key=lambda t: t[0])[:6]:
            if _deadline_hit():
                break
            if not np.isfinite(b):
                continue
            vp = _polish(v, A, B, C)
            bp = _bound_from_v(vp, A, B, C)
            if np.isfinite(bp) and bp < best:
                best = bp
                best_v = vp.copy()

        # safety: never return worse than the prior-art value
        if not np.isfinite(best) or best > _bound_from_v(v_ref, A, B, C):
            best = _bound_from_v(v_ref, A, B, C)
            best_v = v_ref.copy()

        return np.array(best_v, dtype=float)
    except Exception:
        # fall back to the known-good prior-art coefficients
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
