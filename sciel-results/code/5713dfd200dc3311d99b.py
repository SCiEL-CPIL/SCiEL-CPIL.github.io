"""Uncertainty inequality: tune even Hermite-Gaussian to minimize C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# ---------- Gaussian moments: I_k = ∫ x^k e^{-2πx²} dx (k even) ----------
_a = 2.0 * np.pi
_sqrt_pi_a = np.sqrt(np.pi / _a)
MOMENTS = {}
_df = 1.0
for _m in range(16):
    _k = 2 * _m
    if _m == 0:
        _v = _sqrt_pi_a
    else:
        _df *= (2 * _m - 1)
        _v = _df / ((2.0 * _a) ** _m) * _sqrt_pi_a
    MOMENTS[_k] = _v

# ---------- Physicist's Hermite polynomials H_n in monomial (even) basis ----------
# H_COEFFS[n] = [c0, c2, c4, ...] for x^{0}, x^{2}, x^{4}, ...
H_COEFFS = {
    0:  np.array([1.0]),
    4:  np.array([12.0, -48.0, 16.0]),
    8:  np.array([3024.0, -20160.0, 20160.0, -4608.0, 256.0]),
    12: np.array([665280.0, -7983360.0, 831600.0, -7096320.0, 5940.0, -135168.0, 4096.0]),
}
H0 = {0: 1.0, 4: 12.0, 8: 3024.0, 12: 665280.0}

# ---------- Build orthonormal even-parity basis with g(0)=0 ----------
# Vector for coefficient vector (c0,c4,c8,c12): monomial coeffs of the polynomial.
def _hvec(c0, c4, c8, c12):
    v = np.zeros(7)
    v += c0 * H_COEFFS[0]
    v += c4 * H_COEFFS[4]
    v += c8 * H_COEFFS[8]
    v += c12 * H_COEFFS[12]
    return v

def _h0vec(c0, c4, c8, c12):
    return c0 * H0[0] + c4 * H0[4] + c8 * H0[8] + c12 * H0[12]

# Fourier transform of e^{-2πx²}·P(x): even coefficients get sign (-1)^{m} on x^{2m}
def _hvec_f(c0, c4, c8, c12):
    v = _hvec(c0, c4, c8, c12)
    sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    return v * sign

# Constraint g(0)=0 in coefficient space: c0 + 12 c4 + 3024 c8 + 665280 c12 = 0
# We build an orthonormal basis {B0, B1} for the null-space of this row vector.
_row = np.array([1.0, 12.0, 3024.0, 665280.0])
_row = _row / np.linalg.norm(_row)
# Gram-Schmidt: start from e0, project out _row
_e0 = np.array([1.0, 0.0, 0.0, 0.0])
_B0 = _e0 - np.dot(_e0, _row) * _row
_B0 /= np.linalg.norm(_B0)
# second vector orthogonal to both
_e1 = np.array([0.0, 1.0, 0.0, 0.0])
_B1 = _e1 - np.dot(_e1, _row) * _row - np.dot(_e1, _B0) * _B0
_B1 /= np.linalg.norm(_B1)
B = np.column_stack([_B0, _B1])          # shape (4,2); rows are coefficient space
# B.T @ B = I2, and _row @ B = 0  =>  g(0)=0 automatically

# Precompute monomial vectors for the basis (x-space and Fourier-space)
# B_i in coefficient space -> polynomial monomial vector
Bm_x = np.array([_hvec(*B[i]) for i in range(2)])        # (2,7)
Bm_xi = np.array([_hvec_f(*B[i]) for i in range(2)])     # (2,7)

# Inner-product Gram matrix G[j,k] = ∫ (poly_j)(poly_k) e^{-2πx²} dx
def _gram(A):
    n = len(A)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            prod = A[i] * A[j]
            s = 0.0
            for m, c in enumerate(prod):
                if c != 0.0 and (2 * m) in MOMENTS:
                    s += c * MOMENTS[2 * m]
            G[i, j] = G[j, i] = s
    return G

Gx = _gram(Bm_x)
Gxi = _gram(Bm_xi)

def _variance(u, G):
    n = u @ G @ u
    if n <= 0.0:
        return 1e30, 1e30, 0.0
    # ∫ x² u(x)² e^{-2πx²} dx : shift monomial index by 1 (power +2)
    num = 0.0
    L = 7
    for i in range(L):
        for j in range(L):
            c = u[i] * u[j]
            if c == 0.0:
                continue
            # product degree 2*(i+j); multiply by x^2 -> power 2*(i+j)+2
            p = 2 * (i + j) + 2
            if p in MOMENTS:
                num += c * MOMENTS[p]
    return n, num, num / n

def _objective(u):
    """u = (u0,u1) in orthonormal basis. Returns (C4, var_x, var_xi)."""
    nx, mx, vx = _variance(u, Gx)
    if nx <= 0.0:
        return 1e30, 1e30, 1e30
    nxi, mxi, vxi = _variance(u, Gxi)
    if nxi <= 0.0:
        return 1e30, 1e30, 1e30
    return np.sqrt(vx * vxi), vx, vxi

def _grad(u):
    """Analytic gradient of C4 wrt u (2-vector)."""
    nx, mx, vx = _variance(u, Gx)
    nxi, mxi, vxi = _variance(u, Gxi)
    if nx <= 0.0 or nxi <= 0.0:
        return np.zeros(2)
    C = np.sqrt(vx * vxi)
    # dC/du = (vxi/(2C)) d(vx)/du + (vx/(2C)) d(vxi)/du
    # d(vx)/du = 2 (Gx u)/nx - 2 vx (mx u)/nx  ... where (Gx u) is gradient of nx,
    #           (mx u) is gradient of mx (i.e. 2*Gx2*u with Gx2 the x^2-weighted gram)
    # We compute gradients of nx and mx directly.
    def _g_n(u, A):
        return 2.0 * (A @ u)
    def _g_m(u, A):
        # gradient of ∫ x² u² e^{-2πx²}: 2 * (G2 @ u) where G2[i,j]=∫ x² A_i A_j e^{-2πx²}
        g = np.zeros(len(u))
        for i in range(len(u)):
            s = 0.0
            for j in range(len(u)):
                prod = A[i] * A[j]
                t = 0.0
                for m, c in enumerate(prod):
                    if c == 0.0:
                        continue
                    p = 2 * (m + 1)
                    if p in MOMENTS:
                        t += c * MOMENTS[p]
                s += u[j] * t
            g[i] = 2.0 * s
        return g
    gnx = _g_n(u, Gx)
    gmx = _g_m(u, Bm_x)
    gnxi = _g_n(u, Gxi)
    gmxi = _g_m(u, Bm_xi)
    dvx = gnx / nx - vx * gmx / nx
    dvxi = gnxi / nxi - vxi * gmxi / nxi
    return (vxi / (2.0 * C)) * dvx + (vx / (2.0 * C)) * dvxi

def _hess(u):
    """Numerical Hessian (2x2) of C4."""
    h = 1e-6
    H = np.zeros((2, 2))
    g0 = _grad(u)
    for i in range(2):
        ep = np.zeros(2); ep[i] = h
        em = np.zeros(2); em[i] = -h
        for j in range(i, 2):
            epp = np.zeros(2); epp[i] = h; epp[j] = h
            epm = np.zeros(2); epm[i] = h; epm[j] = -h
            emp = np.zeros(2); emp[i] = -h; emp[j] = h
            emm = np.zeros(2); emm[i] = -h; emm[j] = -h
            H[i, j] = (_grad(u + epp)[j] - _grad(u + epm)[j] - _grad(u + emp)[j] + _grad(u + emm)[j]) / (4.0 * h * h)
            H[j, i] = H[i, j]
    return H

# ---------- Seed point (convert original best to basis) ----------
_orig = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_c12 = -(_orig[0] * H0[0] + _orig[1] * H0[4] + _orig[2] * H0[8]) / H0[12]
_coeff = np.array([_orig[0], _orig[1], _orig[2], _c12])
u0 = B.T @ _coeff

def _as_coeff(u):
    """Return (c0, c4, c8) for a basis vector u (c12 derived)."""
    c = B @ u
    return c[0], c[1], c[2]

def construct():
    start = time.time()
    deadline = start + 180.0

    u = u0.copy()
    best = _objective(u)[0]

    def timed_ok():
        return time.time() < deadline - 3.0

    # --- Phase 1: adaptive line search along gradient (robust to scale) ---
    if timed_ok():
        step = 0.05
        for _it in range(40000):
            if not timed_ok():
                break
            g = _grad(u)
            gn = np.linalg.norm(g)
            if gn < 1e-14:
                break
            f = best
            ok = False
            for _ls in range(60):
                if not timed_ok():
                    break
                cand = u - step * g / gn
                fc = _objective(cand)[0]
                if fc < f - 1e-16:
                    u = cand
                    best = fc
                    ok = True
                    break
                step *= 0.5
            if not ok:
                step *= 0.5
                if step < 1e-14:
                    break

    # --- Phase 2: Newton / Hessian polish (exact local solve) ---
    if timed_ok():
        for _it in range(400):
            if not timed_ok():
                break
            g = _grad(u)
            if np.linalg.norm(g) < 1e-14:
                break
            H = _hess(u)
            # regularize
            reg = 1e-12
            try:
                d = np.linalg.solve(H + reg * np.eye(2), -g)
            except np.linalg.LinAlgError:
                d = -g / max(np.linalg.norm(g), 1e-30)
            # backtracking
            f = best
            alpha = 1.0
            ok = False
            for _ls in range(50):
                if not timed_ok():
                    break
                cand = u + alpha * d
                fc = _objective(cand)[0]
                if fc < f - 1e-18:
                    u = cand
                    best = fc
                    ok = True
                    break
                alpha *= 0.5
            if not ok:
                break

    # --- Phase 3: tiny local scan to escape any residual plateau ---
    if timed_ok():
        for scale in [1e-4, 1e-5, 1e-6]:
            if not timed_ok():
                break
            for a in np.linspace(-1, 1, 15):
                for b in np.linspace(-1, 1, 15):
                    if not timed_ok():
                        break
                    cand = u + scale * np.array([a, b])
                    fc = _objective(cand)[0]
                    if fc < best - 1e-18:
                        best = fc
                        u = cand.copy()
            # quick re-polish
            if timed_ok():
                for _it in range(50):
                    if not timed_ok():
                        break
                    g = _grad(u)
                    if np.linalg.norm(g) < 1e-14:
                        break
                    H = _hess(u)
                    try:
                        d = np.linalg.solve(H + 1e-12 * np.eye(2), -g)
                    except np.linalg.LinAlgError:
                        d = -g
                    alpha = 1.0
                    ok = False
                    for _ls in range(40):
                        if not timed_ok():
                            break
                        cand = u + alpha * d
                        fc = _objective(cand)[0]
                        if fc < best - 1e-18:
                            u = cand
                            best = fc
                            ok = True
                            break
                        alpha *= 0.5
                    if not ok:
                        break

    c0, c4, c8 = _as_coeff(u)
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
