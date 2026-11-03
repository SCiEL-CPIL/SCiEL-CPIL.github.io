import numpy as np
from scipy.optimize import minimize
import time


def construct():
    t0 = time.time()
    DL = 175.0  # safety margin before 210s kill

    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 1344 * x**6 + 2688 * x**4 - 1344 * x**2 + 168
    H12 = (4096 * x**12 - 36864 * x**10 + 110592 * x**8
           - 145152 * x**6 + 103680 * x**4 - 30240 * x**2 + 4032)
    gaus = np.exp(-x * x * 0.5)
    x2g = x * x * gaus

    # Precompute basis vectors: g = c0*b0 + c4*b4 + c8*b8
    # where c12 is eliminated via g(0)=0 constraint
    # c12 = -(c0 + 12*c4 + 1680*c8) / 40320
    b0 = (H0 - H12 / 40320) * gaus
    b4 = (H4 - 12 * H12 / 40320) * gaus
    b8 = (H8 - 1680 * H12 / 40320) * gaus

    # Precompute norms and cross terms for the 3 basis vectors
    # norm2 = c^T M c where M is the Gram matrix
    M = np.empty((3, 3))
    for i in range(3):
        bi = [b0, b4, b8][i]
        for j in range(3):
            bj = [b0, b4, b8][j]
            M[i, j] = np.dot(bi, bj) * dx

    # x^2-weighted Gram matrix
    Mx = np.empty((3, 3))
    for i in range(3):
        bi = [b0, b4, b8][i]
        for j in range(3):
            bj = [b0, b4, b8][j]
            Mx[i, j] = np.dot(x2g * bi, bj) * dx

    # Precompute FFT of basis vectors
    Fb0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(b0))) * dx
    Fb4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(b4))) * dx
    Fb8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(b8))) * dx

    # Frequency-weighted cross terms: sum(freqs^2 * Fi * Fj_conj) * dx
    f2 = freqs * freqs
    Fv = [Fb0, Fb4, Fb8]
    Ff = np.empty((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Ff[i, j] = np.dot(f2 * Fv[i], np.conj(Fv[j])) * dx

    def obj3(c):
        """Objective for 3-vector of coefficients."""
        c = np.asarray(c, dtype=float)
        n2 = c @ M @ c
        if n2 <= 1e-30:
            return 1e18
        x2n = c @ Mx @ c
        # Fourier: G = c0*Fb0 + c4*Fb4 + c8*Fb8
        G = c[0] * Fv[0] + c[1] * Fv[1] + c[2] * Fv[2]
        # Fixed: np.dot requires two args. Use np.sum for integral approximation.
        f2n = np.sum(f2 * np.abs(G) ** 2) * dx
        if not np.isfinite(f2n) or f2n < 0:
            return 1e18
        return np.sqrt(x2n / n2 * f2n / n2)

    # The objective is scale-invariant: obj(a*c) = obj(c)
    # So we fix ||c|| = 1 and search on the 2-sphere.
    # Use spherical coordinates: c = [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]
    def obj2(theta_phi):
        th, ph = theta_phi
        c = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        return obj3(c)

    # Structural Change: Use Differential Evolution for robust global optimization
    # This avoids the brittle grid+Nelder-Mead approach and handles non-convexity better.
    from scipy.optimize import differential_evolution

    bounds = [(1e-4, np.pi - 1e-4), (0, 2 * np.pi)]
    
    # Check time before starting expensive global optimization
    if time.time() - t0 > DL - 10:
        # Fallback to a simple heuristic if no time left
        th, ph = np.pi / 2, 0.0
        c = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        return c

    try:
        # DE is stochastic, set seed for reproducibility if needed, but here we just want best result
        # popsize=15 is a good balance for 2D problem
        # maxiter=100 should be sufficient for 2D
        res = differential_evolution(obj2, bounds, maxiter=100, popsize=20, tol=1e-6, seed=42)
        if res.success and np.isfinite(res.fun):
            best_tp = res.x
        else:
            # Fallback to center of domain if DE fails
            best_tp = (np.pi / 2, 0.0)
    except Exception:
        best_tp = (np.pi / 2, 0.0)

    # Final local polish with Nelder-Mead from DE result
    if time.time() - t0 < DL - 5:
        try:
            res_local = minimize(obj2, best_tp, method='Nelder-Mead',
                                 options={'xatol': 1e-10, 'fatol': 1e-12,
                                          'maxiter': 1000, 'adaptive': True})
            if res_local.fun < obj2(best_tp):
                best_tp = res_local.x
        except Exception:
            pass

    th, ph = best_tp
    c = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return c


if __name__ == "__main__":
    print(construct())
