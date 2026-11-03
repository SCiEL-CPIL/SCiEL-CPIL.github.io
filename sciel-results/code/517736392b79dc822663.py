import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Setup grid - use sufficient range and resolution
    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # c12 = -(c0*1 + c4*3 + c8*105) / 10395
    v0 = (H0 - H12 / 10395.0) * gauss
    v4 = (H4 - 3.0 * H12 / 10395.0) * gauss
    v8 = (H8 - 105.0 * H12 / 10395.0) * gauss

    # Precompute Gram matrices for norm, position-weighted, and frequency-weighted
    M = np.zeros((3, 3))
    M[0, 0] = np.sum(v0**2) * dx
    M[1, 1] = np.sum(v4**2) * dx
    M[2, 2] = np.sum(v8**2) * dx
    M[0, 1] = M[1, 0] = np.sum(v0 * v4) * dx
    M[0, 2] = M[2, 0] = np.sum(v0 * v8) * dx
    M[1, 2] = M[2, 1] = np.sum(v4 * v8) * dx

    Mx = np.zeros((3, 3))
    Mx[0, 0] = np.sum(x**2 * v0**2) * dx
    Mx[1, 1] = np.sum(x**2 * v4**2) * dx
    Mx[2, 2] = np.sum(x**2 * v8**2) * dx
    Mx[0, 1] = Mx[1, 0] = np.sum(x**2 * v0 * v4) * dx
    Mx[0, 2] = Mx[2, 0] = np.sum(x**2 * v0 * v8) * dx
    Mx[1, 2] = Mx[2, 1] = np.sum(x**2 * v4 * v8) * dx

    # FFT of basis vectors for frequency-domain computations
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

    Mf = np.zeros((3, 3))
    Mf[0, 0] = np.sum(freqs**2 * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(freqs**2 * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(freqs**2 * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx

    def compute_bound(c):
        """Compute uncertainty product bound δx * δξ for coefficient vector c."""
        c = np.asarray(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 1e-30 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0:
            return np.inf
        if not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    # Seed initial guess
    c_seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_bound = compute_bound(c_seed)
    best_c = c_seed.copy()

    # Phase 1: Global search with differential evolution on the 3D space
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            
            de_result = differential_evolution(
                compute_bound,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=20,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_c = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Multi-start local refinement with Nelder-Mead
    if time.time() - start_time < deadline - 30:
        # Generate multiple starting points around the best found
        rng = np.random.RandomState(42)
        starts = [best_c.copy()]
        for _ in range(10):
            perturbation = rng.normal(0, 0.5, size=3)
            starts.append(best_c + perturbation)
        
        for i, start in enumerate(starts):
            if time.time() - start_time > deadline - 20:
                break
            try:
                res = minimize(
                    compute_bound,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 5000, 'adaptive': True}
                )
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass

    # Phase 3: Final ultra-fine refinement if time permits
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                compute_bound,
                best_c,
                method='Nelder-Mead',
                options={'xatol': 1e-16, 'fatol': 1e-18, 'maxiter': 10000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    return np.array(best_c)
