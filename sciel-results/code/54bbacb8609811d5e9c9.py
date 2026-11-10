# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize, L_BFGS_B


def _build_basis(k=10):
    """Build real orthonormal spherical harmonics Y_m for m=-k..k on the unit sphere."""
    n_m = 2 * k + 1
    m_vals = np.arange(-k, k + 1)

    def N(m):
        return np.sqrt((2 * k + 1) / (4 * np.pi) * np.math.factorial(k - abs(m)) / np.math.factorial(k + abs(m)))

    def Y_m(theta, phi, m):
        P = lpmv(abs(m), k, np.cos(theta))
        if m == 0:
            return N(0) * P
        elif m > 0:
            return np.sqrt(2) * N(m) * P * np.cos(m * phi)
        else:
            return np.sqrt(2) * N(-m) * P * np.sin(-m * phi)

    return m_vals, Y_m, N


def _precompute_tensor(k=10):
    """Precompute C_{m1,m2,m3,m4} = integral Y_{m1} Y_{m2} Y_{m3} Y_{m4} dOmega."""
    m_vals, Y_m, N = _build_basis(k)
    n_m = 2 * k + 1

    # Gauss-Legendre in mu=cos(theta): exact for polynomials of degree < 2*ngl
    ngl = 30
    mu, w = np.polynomial.legendre.leggauss(ngl)
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2 * np.pi / 130

    # basis values: shape (ngl, nphi, n_m)
    Yvals = np.empty((ngl, 130, n_m))
    for j, m in enumerate(m_vals):
        Yvals[:, :, j] = Y_m(mu, phi, m)

    # quadrature weights for dOmega = dmu dphi
    wmu = w * dphi  # (ngl,)

    # Tensor: iterate over m1, m2, m3; for m4 use sum_j Y_j * (Y1*Y2*Y3)
    T = np.zeros((n_m, n_m, n_m, n_m))
    for m1 in range(n_m):
        for m2 in range(n_m):
            for m3 in range(n_m):
                prod = Yvals[:, :, m1] * Yvals[:, :, m2] * Yvals[:, :, m3]  # (ngl, nphi)
                # C[m1,m2,m3,m4] = sum_j prod_j * Yvals_j  -> (n_m,)
                T[m1, m2, m3, :] = np.einsum('ij,ijj->j', wmu[:, None], prod * Yvals)
    return T


def _grad_tensor(c, T):
    """Gradient of c T c c w.r.t. c = 4 * einsum('ijkl,k->ijl', T, c)."""
    return 4.0 * np.einsum('ijkl,k->ijl', T, c)


def _objective(c, T):
    return np.einsum('ijkl,i,j,k,l->', T, c, c, c, c)


def _optimize(T, max_iter=4000):
    n_m = T.shape[0]
    rng = np.random.default_rng(12345)
    best_val = np.inf
    best_c = None

    # Multiple random initializations on the unit sphere
    for trial in range(12):
        if trial == 0:
            c0 = np.ones(n_m, dtype=np.float64)
        else:
            c0 = rng.standard_normal(n_m)
        c0 = c0 / np.linalg.norm(c0)

        def fun(c):
            return _objective(c, T)

        def grad(c):
            return _grad_tensor(c, T)

        res = minimize(fun, c0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
        c = res.x
        c = c / np.linalg.norm(c)
        val = _objective(c, T)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Polishing: a few more L-BFGS-B from the best point
    def fun(c):
        return _objective(c, T)

    def grad(c):
        return _grad_tensor(c, T)

    res = minimize(fun, best_c, jac=grad, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14})
    c = res.x
    c = c / np.linalg.norm(c)
    val = _objective(c, T)
    if val < best_val:
        best_val = val
        best_c = c.copy()

    return best_c


_TENSOR_CACHE = None


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) for a degree-10 spherical harmonic
    f = sum_m c_m Y_m that minimizes ||f||_4 / ||f||_2 on the unit sphere.

    Uses a spectral-domain formulation: the L4^4 norm is a quartic form in the
    coefficients with tensor C_{m1..m4} = integral Y_{m1}Y_{m2}Y_{m3}Y_{m4} dOmega,
    precomputed exactly via Gauss-Legendre quadrature. The homogeneous ratio is
    minimized on the unit sphere using L-BFGS-B with the analytic gradient.

    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m,
            ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    global _TENSOR_CACHE
    if _TENSOR_CACHE is None:
        _TENSOR_CACHE = _precompute_tensor(10)
    T = _TENSOR_CACHE

    c = _optimize(T)
    c = np.asarray(c, dtype=np.float64)

    # Scale so max|c| is comfortably within (1e-3, 1e5). The ratio is scale-invariant.
    scale = 1.0 / np.max(np.abs(c))
    c = c * scale  # now max|c| == 1

    return c


# EVOLVE-BLOCK-END
