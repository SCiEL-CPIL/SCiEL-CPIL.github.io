"""Combined approach: analytic gradient L-BFGS-B on sphere + multi-start + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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

    # Setup grid with higher resolution
    N = 8192
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
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices
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

    def compute_bound_and_grad_from_coeffs(c):
        """Compute bound and analytic gradient for coefficient vector c (assumed unit norm)."""
        c = np.array(c, dtype=float)
        norm2 = c @ c
        if norm2 <= 1e-30 or not np.isfinite(norm2):
            return np.inf, np.zeros(3)
        c_norm = c / np.sqrt(norm2)
        
        x2_val = c_norm @ Mx @ c_norm
        f2_val = c_norm @ Mf @ c_norm
        
        if not np.isfinite(f2_val) or f2_val < 1e-30 or not np.isfinite(x2_val) or x2_val < 1e-30:
            return np.inf, np.zeros(3)
            
        bound = np.sqrt(x2_val * f2_val)
        
        # Gradient of sqrt(x2_val * f2_val) w.r.t. c
        # d(bound)/dc = (1/(2*bound)) * (f2_val * d(x2_val)/dc + x2_val * d(f2_val)/dc)
        # d(x2_val)/dc = 2 * Mx @ c_norm
        # d(f2_val)/dc = 2 * Mf @ c_norm
        # But we need gradient w.r.t. spherical params, so compute gradient w.r.t. c first
        grad_c = (f2_val * (Mx @ c_norm) + x2_val * (Mf @ c_norm)) / bound
        return bound, grad_c

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        bound, _ = compute_bound_and_grad_from_coeffs(c)
        return bound

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    def objective_sphere(theta, phi):
        """Objective for scipy minimize in spherical coordinates."""
        return compute_bound_from_spherical(theta, phi)

    def objective_sphere_grad(theta, phi):
        """Objective and analytic gradient for L-BFGS-B in spherical coordinates."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        
        bound, grad_c = compute_bound_and_grad_from_coeffs(c)
        
        if not np.isfinite(bound):
            return bound, np.array([0.0, 0.0])
        
        # Chain rule: dtheta, dphi
        # c0 = sin(theta)*cos(phi), dc0/dtheta = cos(theta)*cos(phi), dc0/dphi = -sin(theta)*sin(phi)
        # c4 = sin(theta)*sin(phi), dc4/dtheta = cos(theta)*sin(phi), dc4/dphi = sin(theta)*cos(phi)
        # c8 = cos(theta),         dc8/dtheta = -sin(theta),          dc8/dphi = 0
        dtheta = grad_c[0] * np.cos(theta) * np.cos(phi) + grad_c[1] * np.cos(theta) * np.sin(phi) - grad_c[2] * np.sin(theta)
        dphi = -grad_c[0] * np.sin(theta) * np.sin(phi) + grad_c[1] * np.sin(theta) * np.cos(phi)
        
        return bound, np.array([dtheta, dphi])

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(c8)
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Multi-start local optimization with L-BFGS-B (uses analytic gradients)
    if time.time() - start_time < deadline - 30:
        # Generate multiple random starting points
        n_starts = 30
        rng = np.random.RandomState(42)
        
        for i in range(n_starts):
            if time.time() - start_time >= deadline - 25:
                break
                
            # Random spherical coordinates
            theta_rand = rng.uniform(0.05, np.pi - 0.05)
            phi_rand = rng.uniform(0.05, 2 * np.pi - 0.05)
            
            # Try L-BFGS-B from this starting point
            try:
                res = minimize(
                    objective_sphere_grad, 
                    x0=[theta_rand, phi_rand], 
                    method='L-BFGS-B',
                    jac=True,
                    options={'gtol': 1e-10, 'ftol': 1e-12, 'maxiter': 2000, 'maxfun': 5000}
                )
                if res.fun < best_bound and np.isfinite(res.fun):
                    best_bound = res.fun
                    best_theta, best_phi = res.x
            except Exception:
                pass
    
    # Also try from the seed point with tighter tolerance
    if time.time() - start_time < deadline - 20:
        try:
            res = minimize(
                objective_sphere_grad, 
                x0=[best_theta, best_phi], 
                method='L-BFGS-B',
                jac=True,
                options={'gtol': 1e-12, 'ftol': 1e-14, 'maxiter': 5000, 'maxfun': 10000}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Final high-precision polish: use Nelder-Mead with very tight tolerances
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                objective_sphere, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 20000, 'adaptive': True}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
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
