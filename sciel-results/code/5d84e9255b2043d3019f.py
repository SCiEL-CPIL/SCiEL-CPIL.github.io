"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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


def _build_matrices():
    """Build the Gram matrices M, Mx, Mf for the constrained basis."""
    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors with g(0)=0 constraint: c12 = -(c0+12*c4+1680*c8)/40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    M = np.array([
        [np.sum(v0**2) * dx, np.sum(v0 * v4) * dx, np.sum(v0 * v8) * dx],
        [np.sum(v0 * v4) * dx, np.sum(v4**2) * dx, np.sum(v4 * v8) * dx],
        [np.sum(v0 * v8) * dx, np.sum(v4 * v8) * dx, np.sum(v8**2) * dx]
    ])

    Mx = np.array([
        [np.sum(x**2 * v0**2) * dx, np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v0 * v8) * dx],
        [np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v4**2) * dx, np.sum(x**2 * v4 * v8) * dx],
        [np.sum(x**2 * v0 * v8) * dx, np.sum(x**2 * v4 * v8) * dx, np.sum(x**2 * v8**2) * dx]
    ])

    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

    Mf = np.array([
        [np.sum(freqs**2 * np.abs(G0)**2) * dx, np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx, np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx],
        [np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx, np.sum(freqs**2 * np.abs(G4)**2) * dx, np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx],
        [np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx, np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx, np.sum(freqs**2 * np.abs(G8)**2) * dx]
    ])

    return M, Mx, Mf


def _compute_bound(c, M, Mx, Mf):
    """Compute the uncertainty product bound for coefficient vector c."""
    c = np.array(c, dtype=float)
    norm2 = c @ M @ c
    if norm2 <= 0 or not np.isfinite(norm2):
        return np.inf
    x2_norm = c @ Mx @ c
    freq2_norm = c @ Mf @ c
    if not np.isfinite(x2_norm) or x2_norm < 0 or not np.isfinite(freq2_norm) or freq2_norm < 0:
        return np.inf
    return np.sqrt(x2_norm * freq2_norm / (norm2 * norm2))


def _solve_critical_points(M, Mx, Mf):
    """
    Find the minimum of the uncertainty bound using spherical coordinates.
    """
    start_time = time.time()
    deadline_time = _start_time + deadline - 5
    
    def bound_spherical(theta, phi):
        # theta in [0, pi], phi in [0, 2*pi]
        c0 = np.sin(theta) * np.cos(phi)
        c1 = np.sin(theta) * np.sin(phi)
        c2 = np.cos(theta)
        c = np.array([c0, c1, c2])
        return _compute_bound(c, M, Mx, Mf)

    # Use Differential Evolution on the sphere
    try:
        de_result = differential_evolution(
            bound_spherical,
            bounds=[(0, np.pi), (0, 2 * np.pi)],
            seed=42,
            maxiter=300,
            tol=1e-12,
            popsize=30,
            polish=True,
            init='sobol'
        )
        
        best_bound = de_result.fun
        best_theta = de_result.x[0]
        best_phi = de_result.x[1]
        
        c_best = np.array([
            np.sin(best_theta) * np.cos(best_phi),
            np.sin(best_theta) * np.sin(best_phi),
            np.cos(best_theta)
        ])
        
        # Polish with L-BFGS-B on the sphere coordinates
        if time.time() < deadline_time:
            res = minimize(
                bound_spherical,
                x0=de_result.x,
                method='L-BFGS-B',
                bounds=[(0, np.pi), (0, 2 * np.pi)],
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta = res.x[0]
                best_phi = res.x[1]
                c_best = np.array([
                    np.sin(best_theta) * np.cos(best_phi),
                    np.sin(best_theta) * np.sin(best_phi),
                    np.cos(best_theta)
                ])
                
    except Exception:
        c_best = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
        best_bound = _compute_bound(c_best, M, Mx, Mf)
        
    return c_best, best_bound


_start_time = None
deadline = None

def construct():
    global _start_time, deadline
    _start_time = time.time()
    deadline = 175
    
    M, Mx, Mf = _build_matrices()
    
    # Try the exact critical point approach first
    try:
        c_best, best_bound = _solve_critical_points(M, Mx, Mf)
    except Exception:
        # Fallback to seed
        c_best = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
        best_bound = _compute_bound(c_best, M, Mx, Mf)
    
    # Verify and possibly refine
    final_bound = _compute_bound(c_best, M, Mx, Mf)
    
    # If the solution is in a bad region, try the seed
    c_seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_bound = _compute_bound(c_seed, M, Mx, Mf)
    
    if seed_bound < final_bound:
        c_best = c_seed
        final_bound = seed_bound
    
    return c_best

# EVOLVE-BLOCK-END
