# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial


def _build_real_harmonic_basis(k):
    """Real orthonormal degree-k spherical harmonics on a fine grid.
    Index i -> m = i - k.  Grid fine enough to be exact for degree 4k."""
    n_t = 400
    n_p = 400
    theta = np.linspace(0.0, np.pi, n_t)
    phi = np.linspace(0.0, 2.0 * np.pi, n_p)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    cth = np.cos(T)
    Y = np.empty((2 * k + 1, n_t, n_p))
    for i in range(2 * k + 1):
        m = i - k
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                    float(factorial(k - am)) / float(factorial(k + am)))
        Pm = lpmv(am, k, cth)
        if m == 0:
            Y[i] = N * Pm
        elif m > 0:
            Y[i] = np.sqrt(2.0) * N * Pm * np.cos(m * P)
        else:
            Y[i] = np.sqrt(2.0) * N * Pm * np.sin(am * P)
    return Y, theta, phi


def _gaunt_quartic(k):
    """A[a,b,c,d] = (1/4pi) ∫ Y_a Y_b Y_c Y_d dOmega."""
    Y, theta, phi = _build_real_harmonic_basis(k)
    n = Y.shape[0]
    w_theta = np.sin(theta) * (np.pi / (theta.size - 1))
    w_phi = (2.0 * np.pi) / phi.size
    W = np.outer(w_theta, w_phi)
    Yw = Y * W
    A = np.empty((n, n, n, n))
    for a in range(n):
        for b in range(n):
            Yab = Yw[a] * Yw[b]
            for c in range(n):
                Yabc = Yab * Yw[c]
                A[a, b, c, :] = Yabc @ Yw.T
    A *= 1.0 / (4.0 * np.pi)
    return A


def _l4_l2(c, A):
    c = np.asarray(c, dtype=np.float64)
    l2sq = np.dot(c, c)
    if l2sq <= 0.0:
        return np.inf, None
    f4 = np.einsum('abcd,a,b,c,d->', A, c, c, c, c)
    if f4 <= 0.0:
        return np.inf, None
    l4 = f4 ** 0.25
    l2 = np.sqrt(l2sq)
    return l4 / l2, (l4, l2)


def _grad(c, A, l4, l2):
    B = np.einsum('abcd,b,c,d->a', A, c, c, c)
    grad = (l4 ** -3.0) * B
    grad /= l2
    return grad


def _optimize(k, A):
    from scipy.optimize import minimize
    n = 2 * k + 1

    def obj(x):
        v, _ = _l4_l2(x, A)
        return v

    def jac(x):
        v, aux = _l4_l2(x, A)
        if aux is None:
            return np.zeros(n)
        l4, l2 = aux
        return _grad(x, A, l4, l2)

    rng = np.random.default_rng(2024)
    best_x, best_val = None, np.inf
    for restart in range(8):
        x0 = rng.standard_normal(n)
        x0 /= np.linalg.norm(x0)
        res = minimize(obj, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-13,
                                'gtol': 1e-11, 'maxls': 40})
        v, _ = _l4_l2(res.x, A)
        if np.isfinite(v) and v < best_val:
            best_val, best_x = v, res.x.copy()
    if best_x is None:
        best_x = np.ones(n)
    mx = np.max(np.abs(best_x))
    if mx > 0:
        best_x = best_x / mx
    return best_x


def rudin_problem10() -> np.ndarray:
    k = 10
    A = _gaunt_quartic(k)
    coeffs = _optimize(k, A)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    mx = np.max(np.abs(coeffs))
    if not (np.isfinite(mx) and 1e-3 <= mx <= 1e5):
        coeffs = np.ones(2 * k + 1, dtype=np.float64)
    return coeffs


# EVOLVE-BLOCK-END
