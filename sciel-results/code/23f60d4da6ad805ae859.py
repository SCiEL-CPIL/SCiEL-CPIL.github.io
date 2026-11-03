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


def construct():
    start_time = time.time()
    deadline = 180  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
    N = 4096
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
    # c12 = -(c0 + 12c4 + 1680c8) / 40320
    # g = (c0*H0 + c4*H4 + c8*H8 - (c0 + 12c4 + 1680c8)/40320 * H12) * gauss
    # g = c0*(H0 - H12/40320)*gauss + c4*(H4 - 12*H12/40320)*gauss + c8*(H8 - 1680*H12/40320)*gauss
    
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices
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

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    def compute_bound_and_grad_from_spherical(theta, phi):
        """Compute bound and gradient w.r.t. (theta, phi) for L-BFGS-B."""
        s_t = np.sin(theta)
        c_t = np.cos(theta)
        s_p = np.sin(phi)
        c_p = np.cos(phi)
        c = np.array([s_t * c_p, s_t * s_p, c_t])
        
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf, np.zeros(2)
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf, np.zeros(2)
        
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        val = delta_x * delta_xi
        
        # Gradient of log(f) w.r.t. c, then chain rule
        # f = sqrt(x2/norm) * sqrt(freq2/norm)
        # log(f) = 0.5*log(x2) - 0.5*log(norm) + 0.5*log(freq2) - 0.5*log(norm)
        # d(log f)/dc = 0.5*x2_grad/x2 - norm_grad/norm + 0.5*freq2_grad/freq2
        x2_grad = 2 * (Mx @ c)
        norm_grad = 2 * (M @ c)
        freq2_grad = 2 * (Mf @ c)
        
        dlogf_dc = (0.5 * x2_grad / x2_norm - norm_grad / norm2 + 0.5 * freq2_grad / freq2_norm)
        
        # Chain rule: c = (s_t*c_p, s_t*s_p, c_t)
        # dc/dtheta = (c_t*c_p, c_t*s_p, -s_t)
        # dc/dphi = (-s_t*s_p, s_t*c_p, 0)
        dc_dtheta = np.array([c_t * c_p, c_t * s_p, -s_t])
        dc_dphi = np.array([-s_t * s_p, s_t * c_p, 0.0])
        
        grad_theta = val * np.dot(dlogf_dc, dc_dtheta)
        grad_phi = val * np.dot(dlogf_dc, dc_dphi)
        
        return val, np.array([grad_theta, grad_phi])

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    # Convert to spherical for initialization
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(c8)
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Global optimization using Differential Evolution
    # This is a structural change: using a global stochastic optimizer instead of local Nelder-Mead
    # on a 2D parameter space defined by spherical coordinates.
    
    bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
    
    # Check time
    if time.time() - start_time < deadline - 50:
        try:
            # Use a moderate population size for speed
            de_result = differential_evolution(
                compute_bound_from_spherical, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-8,
                popsize=15,
                polish=True
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # Multi-start L-BFGS-B with analytic gradients for high-precision convergence
    if time.time() - start_time < deadline - 30:
        try:
            # Generate multiple starting points around the current best
            starts = [[best_theta, best_phi]]
            # Add perturbations
            rng = np.random.RandomState(42)
            for _ in range(20):
                dtheta = rng.uniform(-0.5, 0.5)
                dphi = rng.uniform(-np.pi, np.pi)
                t = best_theta + dtheta
                p = best_phi + dphi
                t = np.clip(t, 1e-6, np.pi - 1e-6)
                p = p % (2 * np.pi)
                if p < 1e-6:
                    p = 1e-6
                if p > 2 * np.pi - 1e-6:
                    p = 2 * np.pi - 1e-6
                starts.append([t, p])
            
            for start in starts:
                if time.time() - start_time > deadline - 15:
                    break
                try:
                    res = minimize(
                        lambda x: compute_bound_and_grad_from_spherical(x[0], x[1])[0],
                        x0=start,
                        jac=lambda x: compute_bound_and_grad_from_spherical(x[0], x[1])[1],
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                    )
                    if res.fun < best_bound:
                        best_bound = res.fun
                        best_theta, best_phi = res.x
                except Exception:
                    pass
        except Exception:
            pass

    # Final ultra-high-precision polish with tight tolerances
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                lambda x: compute_bound_and_grad_from_spherical(x[0], x[1])[0],
                x0=[best_theta, best_phi],
                jac=lambda x: compute_bound_and_grad_from_spherical(x[0], x[1])[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
