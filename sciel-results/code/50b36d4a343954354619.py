"""Optimize the Hermite-Gaussian test function to reduce the C4 upper bound."""
import time
import numpy as np
from scipy.optimize import minimize

_TIME_LIMIT = 170.0
_T0 = time.time()

# Physicists' Hermite polynomials H_n (monic-ish, H_n = (-1)^n e^{x^2} D^n e^{-x^2})
# We build g(x) = P(x) e^{-x^2/2}, P = a0 H0 + a4 H4 + a8 H8 + a12 H12
# Fourier (unitary, F[f](xi)=1/sqrt(2pi) int f(x)e^{-ixx} dx):
#   F[x^n e^{-x^2/2}](xi) = i^n D^n e^{-xi^2/2}  (since F[e^{-x^2/2}]=e^{-xi^2/2})
# So we express P(x) in monomial basis, transform each x^n term.

# H_n in monomial coefficients (highest power first for np.polyval)
_H0 = np.array([1.0])
_H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
_H8 = np.array([256.0, 0.0, -1344.0, 0.0, 2688.0, 0.0, -1344.0, 0.0, 168.0])
_H12 = np.array([4096.0, 0.0, -46080.0, 0.0, 230400.0, 0.0, -483840.0, 0.0,
                 443520.0, 0.0, -154848.0, 0.0, 10368.0])

# g(0)=0 => a0*H0(0)+a4*H4(0)+a8*H8(0)+a12*H12(0)=0
# H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=10368
_CONST = np.array([1.0, 12.0, 168.0, 10368.0])


def _polyval(coeffs, x):
    return np.polyval(coeffs, x)


def _poly_sum(coeffs_list, weights):
    """Sum weight_i * poly_i as a single monomial coefficient array (highest power first)."""
    maxdeg = max(len(c) - 1 for c in coeffs_list)
    out = np.zeros(maxdeg + 1)
    for c, w in zip(coeffs_list, weights):
        deg = len(c) - 1
        out[:len(c)] += w * c
    return out


def _ft_poly_times_gauss(P_coeffs):
    """
    Given P(x) in monomial coeffs (highest first), return hat_P_coeffs for
    F[P(x) e^{-x^2/2}](xi) / e^{-xi^2/2}, i.e. the polynomial Q such that
    hat_g(xi) = Q(xi) e^{-xi^2/2}.
    F[x^n e^{-x^2/2}](xi) = i^n D^n e^{-xi^2/2}.
    So Q(xi) = sum_k c_k i^k D^k e^{-xi^2/2} / e^{-xi^2/2} ... but D^k e^{-xi^2/2}
    is not a polynomial. Instead, we work with the full function.
    """
    # Return a callable for hat_g(xi) directly.
    def hat_g(xi):
        xi = np.asarray(xi, dtype=float)
        val = np.zeros_like(xi)
        for k, c in enumerate(P_coeffs):
            if c == 0:
                continue
            # i^k D^k e^{-xi^2/2}
            # D^k e^{-xi^2/2} = (-1)^k (2xi)^k ... no, use Hermite relation:
            # D^k e^{-xi^2/2} = (-i)^k ... let's just compute via recurrence numerically.
            # D^k e^{-x^2/2}: use that D e^{-x^2/2} = -x e^{-x^2/2}
            # D^k e^{-x^2/2} = (-1)^k He_k(x) e^{-x^2/2} where He_k are probabilists' Hermite
            # But we need explicit; compute D^k e^{-x^2/2} = H_k^{prob}(x) e^{-x^2/2} * (-1)^k
            # Simpler: use the fact that D^k e^{-x^2/2} can be built via recurrence.
            val += c * (1j) ** k * _dkg(xi, k)
        return val
    return hat_g


def _dkg(xi, k):
    """D^k e^{-xi^2/2} computed via probabilists' Hermite: D^k e^{-x^2/2}=(-1)^k He_k(x) e^{-x^2/2}."""
    # He_k (probabilists): He_0=1, He_1=x, He_{n}=x He_{n-1} - (n-1) He_{n-2}
    xi = np.asarray(xi, dtype=float)
    e = np.exp(-0.5 * xi * xi)
    if k == 0:
        return e
    if k == 1:
        return -xi * e
    He_prev2 = np.ones_like(xi)   # He_0
    He_prev1 = xi.copy()          # He_1
    for n in range(2, k + 1):
        He_n = xi * He_prev1 - (n - 1) * He_prev2
        He_prev2, He_prev1 = He_prev1, He_n
    return ((-1.0) ** k) * He_prev1 * e


def _build_g_and_hatg(triplet):
    a0, a4, a8 = triplet
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 10368.0
    weights = np.array([a0, a4, a8, a12])
    # g(x) = P(x) e^{-x^2/2}
    def g(x):
        x = np.asarray(x, dtype=float)
        P = (a0 * _polyval(_H0, x) + a4 * _polyval(_H4, x) +
             a8 * _polyval(_H8, x) + a12 * _polyval(_H12, x))
        return P * np.exp(-0.5 * x * x)
    # hat_g(xi): P in monomial basis
    Pmon = _poly_sum([_H0, _H4, _H8, _H12], weights)
    # Pmon is highest-power-first; convert to ascending for per-degree transform
    Pasc = Pmon[::-1]  # Pasc[k] = coeff of x^k
    def hat_g(xi):
        xi = np.asarray(xi, dtype=float)
        val = np.zeros_like(xi)
        for k in range(len(Pasc)):
            c = Pasc[k]
            if c == 0:
                continue
            val += c * (1j) ** k * _dkg(xi, k)
        return val
    return g, hat_g


def _objective(triplet, npts=20001, xmax=12.0):
    a0, a4, a8 = triplet
    # penalty for g(0) != 0 (should be ~0 by construction)
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 10368.0
    g, hat_g = _build_g_and_hatg(triplet)
    x = np.linspace(-xmax, xmax, npts)
    gv = g(x)
    hv = hat_g(x)
    # C4 bound = sup_x Re[ g(x) * conj(hat_g(-x)) ] / |hat_g(x)|^2  (a common form)
    # We'll use sup of |g(x)|^2 / |hat_g(x)|^2 as the uncertainty bound proxy.
    denom = np.abs(hv) ** 2
    safe = np.where(denom > 1e-300, denom, np.inf)
    ratio = np.abs(gv) ** 2 / safe
    ratio = np.where(denom < 1e-15, 1e6, ratio)
    return float(np.max(ratio))


def construct():
    deadline = _T0 + _TIME_LIMIT
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_p = seed.copy()
    best_val = _objective(seed)

    starts = [seed.copy()]
    rng = np.random.default_rng(0)
    for _ in range(10):
        starts.append(rng.normal(seed, [0.1, 0.015, 0.00025]))
    # deterministic perturbations
    for dx, dy, dz in [(-0.1, 0.01, 0.0), (0.1, -0.01, 0.0), (0.0, 0.02, 0.0),
                       (-0.05, -0.01, 0.0001), (0.05, 0.01, -0.0001),
                       (-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)]:
        starts.append(seed + np.array([dx, dy, dz]))

    for t0 in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(_objective, t0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-11,
                                    'adaptive': True})
            t1 = res.x
            val = _objective(t1)
            if val < best_val:
                best_val = val
                best_p = t1.copy()
        except Exception:
            continue

    # polish
    try:
        if time.time() < deadline - 15.0:
            res2 = minimize(_objective, best_p, method='Nelder-Mead',
                            options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-13,
                                     'adaptive': True})
            t2 = res2.x
            val2 = _objective(t2)
            if val2 < best_val:
                best_val = val2
                best_p = t2.copy()
    except Exception:
        pass

    return np.array([best_p[0], best_p[1], best_p[2]])
