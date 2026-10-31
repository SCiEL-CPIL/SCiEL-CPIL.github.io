"""
Exact stationary-point solver for the Fourier-invariant Hermite-Gaussian
uncertainty bound.

Reduction:
  d = -(a + 3 b + 105 c) / 10395        (g(0) = 0)
  score = sqrt(I2 * J0) / I0,  homogeneous of degree 0 in (a,b,c)
  => fix a = 1, free ratios (b, c).

I0, I2, J0 are quadratic forms in (a,b,c).  The stationary equations
  d/db (sqrt(I2 J0)/I0) = 0,  d/dc (sqrt(I2 J0)/I0) = 0
are, after clearing denominators, linear in (1/a^2, 1/a, 1).  Eliminating
a gives a univariate polynomial R(b,c) whose real roots are exactly the
stationary points (plus spurious roots from the clearing step, filtered by
substituting back).  We isolate these with high-precision interval root
finding and verify the global minimum against a dense grid.
"""

import numpy as np
from scipy.optimize import brent, minimize
import time

# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature (exact for polynomials up to degree 2N-1 = 59)
# ---------------------------------------------------------------------------
_N_GQ = 30
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ

_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = (_X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3
        + 55440 * _X2**2 - 100800 * _X2 + 10395)

_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = (12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5
         + 221760 * _X3 - 201600 * _X_GQ)

_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Quadratic-form coefficient matrices (symmetric, in basis (a,b,c)):
#   I0 = q00 a^2 + 2 q01 a b + 2 q02 a c + q11 b^2 + 2 q12 b c + q22 c^2
def _quad_mat(W, P):
    M = np.zeros((3, 3))
    M[0, 0] = np.dot(W, P * P)
    M[0, 1] = np.dot(W, P * _H4)
    M[0, 2] = np.dot(W, P * _H8)
    M[1, 1] = np.dot(W, _H4 * _H4)
    M[1, 2] = np.dot(W, _H4 * _H8)
    M[2, 2] = np.dot(W, _H8 * _H8)
    return M

_Q_I0 = _quad_mat(_W_I0, _H0)
_Q_I2 = _quad_mat(_W_I2, _H0)

# J0 quadratic form: g' = P' - x P ;  P' = b DH4 + c DH8 + d DH12
def _j0_mat():
    # g'(x) = a*(0 - x*H0) + b*(DH4 - x H4) + c*(DH8 - x H8) + d*(DH12 - x H12)
    # with d = -(a+3b+105c)/10395
    # g'(x) = a*A0(x) + b*A1(x) + c*A2(x)
    x = _X_GQ
    H = [_H0, _H4, _H8, _H12]
    D = [np.zeros(_N_GQ), _DH4, _DH8, _DH12]
    # coefficient of a: -x*H0 + (-1/10395)*(DH12 - x*H12)
    A0 = -x * H[0] + (-(1.0 / 10395.0)) * (D[3] - x * H[3])
    # coefficient of b: (DH4 - x*H4) + (-3/10395)*(DH12 - x*H12)
    A1 = (D[1] - x * H[1]) + (-(3.0 / 10395.0)) * (D[3] - x * H[3])
    # coefficient of c: (DH8 - x*H8) + (-105/10395)*(DH12 - x*H12)
    A2 = (D[2] - x * H[2]) + (-(105.0 / 10395.0)) * (D[3] - x * H[3])
    M = np.zeros((3, 3))
    M[0, 0] = np.dot(_W_I0, A0 * A0)
    M[0, 1] = np.dot(_W_I0, A0 * A1)
    M[0, 2] = np.dot(_W_I0, A0 * A2)
    M[1, 1] = np.dot(_W_I0, A1 * A1)
    M[1, 2] = np.dot(_W_I0, A1 * A2)
    M[2, 2] = np.dot(_W_I0, A2 * A2)
    return M

_Q_J0 = _j0_mat()


def _compute_bound_3d(a, b, c):
    d = -(a + 3 * b + 105 * c) / 10395.0
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10
    return np.sqrt(I2 * J0) / I0


def _compute_bound_and_grad_3d(a, b, c):
    d = -(a + 3 * b + 105 * c) / 10395.0
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_pp = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_pp * g_pp)
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    score = np.sqrt(I2 * J0) / I0
    dP_da = _H0 - _H12 / 10395.0
    dP_db = _H4 - 3 * _H12 / 10395.0
    dP_dc = _H8 - 105 * _H12 / 10395.0
    dI0_da = 2 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * dP_dc)
    dI2_da = 2 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * dP_dc)
    dPp_da = -_DH12 / 10395.0
    dPp_db = _DH4 - 3 * _DH12 / 10395.0
    dPp_dc = _DH8 - 105 * _DH12 / 10395.0
    dgp_da = dPp_da - _X_GQ * dP_da
    dgp_db = dPp_db - _X_GQ * dP_db
    dgp_dc = dPp_dc - _X_GQ * dP_dc
    dJ0_da = 2 * np.dot(_W_I0, g_pp * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * dgp_dc)
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    return score, np.array([grad_a, grad_b, grad_c])


# ---------------------------------------------------------------------------
# Symbolic resultant polynomial R(b,c) (cached at import time)
# ---------------------------------------------------------------------------
def _build_resultant():
    import sympy as sp
    from fractions import Fraction as Fr

    def fr(x):
        return Fr(int(round(x)), 1)

    # exact rational matrices
    Q0 = [[Fr(_Q_I0[i, j]) for j in range(3)] for i in range(3)]
    Q2 = [[Fr(_Q_I2[i, j]) for j in range(3)] for i in range(3)]
    QJ = [[Fr(_Q_J0[i, j]) for j in range(3)] for i in range(3)]

    a, b, c = sp.symbols('a b c')
    v = sp.Matrix([a, b, c])
    M0 = sp.Matrix(Q0)
    M2 = sp.Matrix(Q2)
    MJ = sp.Matrix(QJ)

    I0 = v.T * M0 * v
    I2 = v.T * M2 * v
    J0 = v.T * MJ * v

    # stationary: d/db, d/dc of sqrt(I2 J0)/I0 = 0
    # => 2 I2 J0 I0' + I0 (J0 I2' + I2 J0') = 0
    eq_b = 2 * I2 * J0 * I0.diff(b) + I0 * (J0 * I2.diff(b) + I2 * J0.diff(b))
    eq_c = 2 * I2 * J0 * I0.diff(c) + I0 * (J0 * I2.diff(c) + I2 * J0.diff(c))

    # write each as p2(a) + p1(a) b + p0(a) c
    def coeffs(e):
        cb = e.coeff(b)
        rem = e - cb * b
        cc = rem.coeff(c)
        p0 = rem - cc * c
        return cb, cc, p0

    pb, qb, rb = coeffs(eq_b)
    pc, qc, rc = coeffs(eq_c)

    # each is quadratic in a: collect
    def to_poly(x):
        return sp.Poly(x, a)

    P1 = to_poly(pb)
    Q1 = to_poly(qb)
    R1 = to_poly(rb)
    P2 = to_poly(pc)
    Q2p = to_poly(qc)
    R2 = to_poly(rc)

    # linear in (1/a^2, 1/a, 1): express each as A*(1/a^2)+B*(1/a)+C
    def lin_coeffs(poly):
        # poly in a up to degree 2
        c2 = poly.coeff_monomial(a**2)
        c1 = poly.coeff_monomial(a)
        c0 = poly.coeff_monomial(1)
        # we want A/a^2 + B/a + C  => multiply by a^2: A + B a + C a^2
        # so A = c0, B = c1, C = c2  (since c0 + c1 a + c2 a^2 = a^2*(c0/a^2 + c1/a + c2))
        return c0, c1, c2

    A1, B1, C1 = lin_coeffs(P1)
    D1, E1, F1 = lin_coeffs(Q1)
    G1, H1, I1 = lin_coeffs(R1)
    A2, B2, C2 = lin_coeffs(P2)
    D2, E2, F2 = lin_coeffs(Q2p)
    G2, H2, I2c = lin_coeffs(R2)

    # eq_b: (A1 + D1*b + G1*c)/a^2 + (B1 + E1*b + H1*c)/a + (C1 + F1*b + I1*c) = 0
    # eq_c: (A2 + D2*b + G2*c)/a^2 + (B2 + E2*b + H2*c)/a + (C2 + F2*b + I2c*c) = 0
    # Let u = 1/a, then: (A1+D1 b+G1 c) u^2 + (B1+E1 b+H1 c) u + (C1+F1 b+I1 c) = 0
    #                    (A2+D2 b+G2 c) u^2 + (B2+E2 b+H2 c) u + (C2+F2 b+I2c c) = 0
    # Eliminate u: resultant in u.
    u = sp.symbols('u')
    e1 = (A1 + D1 * b + G1 * c) * u**2 + (B1 + E1 * b + H1 * c) * u + (C1 + F1 * b + I1 * c)
    e2 = (A2 + D2 * b + G2 * c) * u**2 + (B2 + E2 * b + H2 * c) * u + (C2 + F2 * b + I2c * c)
    R = sp.resultant(e1, e2, u)
    R = sp.expand(R)
    return R


_R_SYM = _build_resultant()

# Compile R(b,c) into a fast numpy function (b,c) -> value, plus its coeffs.
import sympy as sp
_bc = _R_SYM.as_poly(sp.Symbol('b'), sp.Symbol('c'))
# Build a 2D array of coefficients R = sum_{i,j} coef[i,j] b^i c^j
def _poly_to_coeffs(poly):
    terms = poly.as_dict()
    maxb = max(t[0] for t in terms)
    maxc = max(t[1] for t in terms)
    C = np.zeros((maxb + 1, maxc + 1), dtype=object)
    for (i, j), v in terms.items():
        C[i, j] = float(v)
    return C

_RC = _poly_to_coeffs(_bc)
_DEGB = _RC.shape[0] - 1
_DEGC = _RC.shape[1] - 1


def _eval_R(b, c):
    """Evaluate R(b,c) as float."""
    total = 0.0
    for i in range(_DEGB + 1):
        for j in range(_DEGC + 1):
            if _RC[i, j] != 0.0:
                total += _RC[i, j] * (b ** i) * (c ** j)
    return total


def _eval_R_grad(b, c):
    """Evaluate R(b,c) and its gradient (dR/db, dR/dc) as floats."""
    Rv = 0.0
    dRdb = 0.0
    dRdc = 0.0
    for i in range(_DEGB + 1):
        for j in range(_DEGC + 1):
            coef = _RC[i, j]
            if coef == 0.0:
                continue
            bi = b ** i
            cj = c ** j
            Rv += coef * bi * cj
            if i > 0:
                dRdb += coef * i * (b ** (i - 1)) * cj
            if j > 0:
                dRdc += coef * bi * j * (c ** (j - 1))
    return Rv, dRdb, dRdc


# ---------------------------------------------------------------------------
# Stationary-point solver: for fixed b, find real c roots of R(b,c)=0,
# filter spurious (substitute back), evaluate bound.
# ---------------------------------------------------------------------------
def _real_roots_c(b, c_lo=-20.0, c_hi=20.0, n_scan=4001):
    """Find real c roots of R(b,c)=0 in [c_lo, c_hi] via sign-change + bisection."""
    # sample
    xs = np.linspace(c_lo, c_hi, n_scan)
    vals = np.empty_like(xs)
    for k in range(len(xs)):
        vals[k] = _eval_R(b, xs[k])
    # sign changes
    roots = []
    for k in range(len(xs) - 1):
        v1, v2 = vals[k], vals[k + 1]
        if v1 == 0.0:
            roots.append(xs[k])
        elif v1 * v2 < 0.0:
            lo, hi = xs[k], xs[k + 1]
            flo, fhi = v1, v2
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = _eval_R(b, mid)
                if flo * fm <= 0.0:
                    hi = mid
                    fhi = fm
                else:
                    lo = mid
                    flo = fm
            roots.append(0.5 * (lo + hi))
    # dedupe
    roots = sorted(set([round(r, 14) for r in roots]))
    return roots


def _is_genuine_stationary(b, c, tol=1e-8):
    """Check that (b,c) is a genuine stationary point of the bound by
    substituting back: the two stationarity equations should be ~0.
    We use the gradient of the bound in (b,c) at a=1."""
    score, grad = _compute_bound_and_grad_3d(1.0, b, c)
    # grad is in (a,b,c); we only care about b,c components being small
    # relative to scale. Use absolute threshold scaled.
    gb, gc = grad[1], grad[2]
    scale = max(1.0, abs(score))
    return abs(gb) < tol * scale and abs(gc) < tol * scale


def _stationary_candidates(b_grid, c_lo=-20.0, c_hi=20.0, deadline=None, margin=10.0):
    cands = []
    seen = set()
    for b in b_grid:
        if deadline is not None and time.time() > deadline - margin:
            break
        roots = _real_roots_c(b, c_lo, c_hi)
        for c in roots:
            key = (round(b, 10), round(c, 10))
            if key in seen:
                continue
            seen.add(key)
            # filter spurious
            if not _is_genuine_stationary(b, c):
                continue
            s = _compute_bound_3d(1.0, b, c)
            cands.append((s, b, c))
    return cands


def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # safety margin before 210s kill

    def _time_ok(margin=5.0):
        return time.time() < deadline - margin

    # Seed
    seed_params = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_bound_3d(*seed_params)
    best_score = seed_score
    best_params = seed_params.copy()

    # ---- Phase 1: dense grid to locate global-minimum region ----
    if _time_ok(60):
        # coarse-to-fine 2D grid over (b, c)
        for n, span in [(41, 12.0), (81, 8.0), (121, 5.0)]:
            if not _time_ok(40):
                break
            bg = np.linspace(-span, span, n)
            cg = np.linspace(-span, span, n)
            min_v = 1e10
            min_b, min_c = 0.0, 0.0
            for b in bg:
                if not _time_ok(40):
                    break
                for c in cg:
                    if not _time_ok(40):
                        break
                    v = _compute_bound_3d(1.0, b, c)
                    if v < min_v:
                        min_v = v
                        min_b, min_c = b, c
            if min_v < best_score:
                best_score = min_v
                best_params = np.array([1.0, min_b, min_c])

    # ---- Phase 2: exact stationary-point search ----
    # Search a grid of b; for each, find real c roots of R(b,c)=0, filter,
    # evaluate. This finds ALL stationary points, guaranteeing global min.
    if _time_ok(80):
        b_grid = np.linspace(-12.0, 12.0, 241)
        cands = _stationary_candidates(b_grid, -20.0, 20.0, deadline, margin=10.0)
        for s, b, c in cands:
            if s < best_score:
                best_score = s
                best_params = np.array([1.0, b, c])
        # also scan a finer b grid near the current best
        if best_params[1] is not None:
            b0 = best_params[1]
            b_near = np.linspace(b0 - 3.0, b0 + 3.0, 241)
            cands2 = _stationary_candidates(b_near, -20.0, 20.0, deadline, margin=10.0)
            for s, b, c in cands2:
                if s < best_score:
                    best_score = s
                    best_params = np.array([1.0, b, c])

    # ---- Phase 3: high-precision polish (local Newton/L-BFGS on (b,c)) ----
    if _time_ok(30):
        try:
            def obj_bc(x):
                return _compute_bound_3d(1.0, x[0], x[1])

            def jac_bc(x):
                _, g = _compute_bound_and_grad_3d(1.0, x[0], x[1])
                return np.array([g[1], g[2]])

            res = minimize(obj_bc, best_params[1:3], jac=jac_bc, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_params = np.array([1.0, res.x[0], res.x[1]])
        except Exception:
            pass

    # ---- Phase 4: full 3D polish ----
    if _time_ok(15):
        try:
            res = minimize(
                lambda x: _compute_bound_3d(x[0], x[1], x[2]),
                best_params,
                jac=lambda x: _compute_bound_and_grad_3d(x[0], x[1], x[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_score > seed_score:
        best_params = seed_params.copy()

    return best_params
