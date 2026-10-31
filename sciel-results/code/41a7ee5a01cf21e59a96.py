"""Combined: exact moment-based bound (inspiration 1) + L-BFGS-B multi-start polish (current)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 210.0
_MARGIN = 12.0

# I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(30)
I[0] = np.sqrt(np.pi)
for n in range(1, 30):
    I[n] = I[n - 1] * (2.0 * n - 1.0) / (2.0 * n)

# Power-basis coefficients (order x^0, x^2, ..., x^12)
H0_c = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
H4_c = np.array([3, -6, 1, 0, 0, 0, 0], dtype=float)
H8_c = np.array([105, -420, 210, -28, 1, 0, 0], dtype=float)
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1], dtype=float)

# a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0, with values at x=0:
# H0_0=1, H4_0=-12, H8_0=168, H12_0=945  (physicist's Hermite)
# => a12 = -(a0 -12 a4 + 168 a8)/945 = (-a0 + 12 a4 - 168 a8)/945
c12 = np.array([-1.0 / 945.0, 12.0 / 945.0, -168.0 / 945.0], dtype=float)

# p = A @ a  (7 x 3)
col0 = H0_c + c12[0] * H12_c
col1 = H4_c + c12[1] * H12_c
col2 = H8_c + c12[2] * H12_c
A_mat = np.column_stack((col0, col1, col2))

# q_k = p_k * (-1)^k
_SGN = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)


def _get_p(a):
    return A_mat @ a


def _moments(p, q):
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    for i in range(7):
        pi = p[i]
        qi = q[i]
        for j in range(7):
            Mx0 += pi * p[j] * I[i + j]
            Mx1 += pi * p[j] * I[i + j + 1]
            Mxi0 += qi * q[j] * I[i + j]
            Mxi1 += qi * q[j] * I[i + j + 1]
    return Mx0, Mx1, Mxi0, Mxi1


def _compute_bound_and_grad(a):
    p = _get_p(a)
    q = p * _SGN
    Mx0, Mx1, Mxi0, Mxi1 = _moments(p, q)

    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)

    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0

    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)

    bound = np.sqrt(vx * vxi)
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)

    # S[n, i] = sum_j p_j I[i+j+n]  and same for q
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i + n:i + n + 7])
            Sxi[n, i] = np.dot(q, I[i + n:i + n + 7])

    # dM[n, m] = 2 * sum_i A[i,m] S[n,i]
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])

    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    Mx0sq = Mx0 * Mx0
    Mxi0sq = Mxi0 * Mxi0
    for m in range(3):
        dvx[m] = (dMx[1, m] * Mx0 - Mx1 * dMx[0, m]) / Mx0sq
        dvxi[m] = (dMxi[1, m] * Mxi0 - Mxi1 * dMxi[0, m]) / Mxi0sq

    factor = 1.0 / (2.0 * bound)
    grad = factor * (vxi * dvx + vx * dvxi)

    return bound, grad


def _objective_and_grad(a):
    return _compute_bound_and_grad(a)


def _safe_minimize(x0, maxiter, maxfun, ftol, gtol):
    return minimize(
        _objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol},
    )


def construct():
    """Return the three Hermite coefficients (degrees 0, 4, 8)."""
    seed_params = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240], dtype=float)

    best_val, _ = _compute_bound_and_grad(seed_params)
    best_params = seed_params.copy()

    def _consider(res):
        nonlocal best_val, best_params
        if res is not None and np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()

    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    starts = [seed_params.copy()]
    for _ in range(16):
        starts.append(seed_params + rng.normal(0, 0.15, size=3))
    for scale in [0.5, 1.5, 2.0, 0.25]:
        starts.append(seed_params * scale)
    # Broad box starts
    for _ in range(8):
        starts.append(rng.uniform(-2.0, 0.5, size=3))

    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            _consider(_safe_minimize(x0, 300, 600, 1e-12, 1e-10))
        except Exception:
            continue

    # Final high-precision polish passes
    for _ in range(2):
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            _consider(_safe_minimize(best_params, 2000, 5000, 1e-15, 1e-13))
        except Exception:
            break

    if not np.all(np.isfinite(best_params)):
        return seed_params.copy()
    return best_params
# EVOLVE-BLOCK-END
