"""Optimized program: numerically minimize the derived upper bound over the
three free Hermite coefficients (degrees 0, 4, 8); the degree-12 coefficient
is fixed so that g(0) = 0. Uses scipy to optimize the natural test-function
bound C = sqrt(int|g|^4) * sqrt(int|g_hat|^4) / int|g|^2."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial
from scipy.integrate import quad


def _hermite_combo(a0, a1, a2):
    """Return coefficients for H0, H4, H8, H12 with g(0)=0 (physicists' H)."""
    # H_n(0) for even n (physicists): H0=1, H4=-12, H8=1680, H12=-2075040
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -2075040.0
    # a0*H0(0)+a1*H4(0)+a2*H8(0)+a3*H12(0)=0  =>  a3 = -(a0*H0_0 + a1*H4_0 + a2*H8_0)/H12_0
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    return np.array([a0, a1, a2, a3])


def _build_g(x, coeffs):
    a0, a1, a2, a3 = coeffs
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    poly = a0 * H0 + a1 * H4 + a2 * H8 + a3 * H12
    return np.exp(-0.5 * x * x) * poly

def _ft_gaussian_hermite(coeffs):
    """Return a function G(k) = Fourier transform of g(x) = e^{-x^2/2} * sum_j a_j H_{n_j}(x),
    using the convention G(k) = int g(x) e^{-i k x} dx.
    For g(x) = e^{-x^2/2} H_n(x), FT is (-i)^n H_n(k) e^{-k^2/2} * sqrt(pi)?? Let's derive:
    Standard: FT[e^{-x^2/2} H_n(x)] = (-i)^n H_n(k) e^{-k^2/2} * sqrt(2*pi)?? 
    Actually, with convention G(k)=int g(x) e^{-i k x} dx:
    FT[e^{-x^2/2}] = sqrt(2*pi) e^{-k^2/2}
    FT[e^{-x^2/2} H_n(x)] = (-i)^n sqrt(2*pi) H_n(k) e^{-k^2/2}
    """
    degrees = [0, 4, 8, 12]
    def G(k):
        total = 0.0
        for a, n in zip(coeffs, degrees):
            if a == 0.0:
                continue
            # (-i)^n for even n: (-i)^n = (-1)^{n/2}
            phase = (-1.0) ** (n // 2)
            Hn_k = eval_hermite(n, k)
            total += a * phase * Hn_k
        return total * np.sqrt(2.0 * np.pi) * np.exp(-0.5 * k * k)
    return G


def _bound(coeffs):
    # Compute int |g|^2 analytically: for g = e^{-x^2/2} P(x) with P polynomial,
    # int e^{-x^2} P(x)^2 dx. Use Gauss-Hermite quadrature for speed and accuracy.
    # But simpler: use quad for all three integrals; the functions are smooth and decay fast.
    
    a0, a1, a2, a3 = coeffs
    if max(abs(a0), abs(a1), abs(a2), abs(a3)) < 1e-15:
        return 1e18
    
    def g_abs4(x):
        gx = _build_g(np.array([x]), coeffs)[0]
        return abs(gx) ** 4
    
    def G_abs4(k):
        Gk = _ft_gaussian_hermite(coeffs)(np.array([k]))[0]
        return abs(Gk) ** 4
    
    # int |g|^4 dx
    int_g4, _ = quad(g_abs4, -15.0, 15.0, limit=200)
    # int |G|^4 dk
    int_G4, _ = quad(G_abs4, -15.0, 15.0, limit=200)
    # int |g|^2 dx: use analytic or quad
    def g_abs2(x):
        gx = _build_g(np.array([x]), coeffs)[0]
        return abs(gx) ** 2
    int_g2, _ = quad(g_abs2, -15.0, 15.0, limit=200)
    
    if int_g2 <= 1e-15:
        return 1e18
    return np.sqrt(int_g4) * np.sqrt(int_G4) / int_g2


def construct():
    # Multistart optimization of the three free coefficients.
    best = None
    best_val = np.inf
    rng = np.random.default_rng(0)
    starts = [
        [-113 / 100.0, 1 / 25.0, 1 / 3240.0],  # literature seed
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.5, 0.1],
        [0.5, -0.5, 0.2],
        [0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1],
        [2.0, -1.0, 0.5],
        [-2.0, 1.0, -0.5],
        [0.3, 0.7, -0.2],
        [rng.normal(size=3) for _ in range(12)],
    ]
    for s in starts:
        s = np.asarray(s, dtype=float)
        try:
            res = minimize(
                _bound,
                s,
                method="Nelder-Mead",
                options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True},
            )
            val = _bound(res.x)
            if val < best_val:
                best_val = val
                best = res.x
        except Exception:
            continue
    if best is None:
        return np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
    # Refine with a tighter local search around the best.
    try:
        res2 = minimize(
            _bound,
            best,
            method="Powell",
            options={"maxiter": 5000, "ftol": 1e-14},
        )
        val2 = _bound(res2.x)
        if val2 < best_val:
            best_val = val2
            best = res2.x
    except Exception:
        pass
    # One more Nelder-Mead refinement from the Powell result
    try:
        res3 = minimize(
            _bound,
            best,
            method="Nelder-Mead",
            options={"maxiter": 8000, "xatol": 1e-10, "fatol": 1e-14, "adaptive": True},
        )
        val3 = _bound(res3.x)
        if val3 < best_val:
            best_val = val3
            best = res3.x
    except Exception:
        pass
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
