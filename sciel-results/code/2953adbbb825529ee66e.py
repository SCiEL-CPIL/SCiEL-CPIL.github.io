"""Optimized program: numerically minimize the derived upper bound over the
three free Hermite coefficients (degrees 0, 4, 8); the degree-12 coefficient
is fixed so that g(0) = 0. Uses scipy to optimize the natural test-function
bound C = sqrt(int|g|^4) * sqrt(int|g_hat|^4) / int|g|^2."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite


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


def _bound(coeffs):
    x = np.linspace(-30.0, 30.0, 40001)
    g = _build_g(x, coeffs)
    # Fourier transform via FFT on a fine grid (use a larger zero-padded grid)
    # Use direct numerical FT via scipy for accuracy on the support.
    from scipy.fft import fft, fftshift, fftfreq
    N = 1 << 16  # 65536
    # Re-grid g onto a uniform grid suitable for FFT
    L = 120.0
    xn = np.linspace(-L / 2, L / 2, N, endpoint=False)
    gn = np.interp(xn, x, g)
    dk = 2.0 * np.pi / (N * (xn[1] - xn[0]))
    G = fftshift(fft(gn)) * (xn[1] - xn[0])
    k = fftshift(fftfreq(N, d=(xn[1] - xn[0]))) * 2.0 * np.pi
    int_g4 = np.trapz(np.abs(gn) ** 4, xn)
    int_G4 = np.trapz(np.abs(G) ** 4, k)
    int_g2 = np.trapz(np.abs(gn) ** 2, xn)
    if int_g2 <= 0:
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
        [rng.normal(size=3) for _ in range(8)],
    ]
    for s in starts:
        s = np.asarray(s, dtype=float)
        try:
            res = minimize(
                _bound,
                s,
                method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10},
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
            options={"maxiter": 3000, "ftol": 1e-12},
        )
        if _bound(res2.x) < best_val:
            best = res2.x
    except Exception:
        pass
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
