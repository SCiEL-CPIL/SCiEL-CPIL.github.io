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
        """Compute bound and analytic gradient for spherical parameters."""
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        
        c0 = st * cp
        c4 = st * sp
        c8 = ct
        c_vec = np.array([c0, c4, c8])
        
        norm2 = c_vec @ M @ c_vec
        
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf, np.array([0.0, 0.0])
        
        x2_norm = c_vec @ Mx @ c_vec
        freq2_norm = c_vec @ Mf @ c_vec
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf, np.array([0.0, 0.0])
        
        A = x2_norm / norm2
        B = freq2_norm / norm2
        
        if A <= 0 or B <= 0:
            return np.inf, np.array([0.0, 0.0])
        
        bound = np.sqrt(A * B)
        
        # dA/dc = (Mx*c)/n2 - x2*M*c/n2^2
        dA_dc = (Mx @ c_vec) / norm2 - (x2_norm / norm2**2) * (M @ c_vec)
        dB_dc = (Mf @ c_vec) / norm2 - (freq2_norm / norm2**2) * (M @ c_vec)
        
        # df/dc = 0.5*sqrt(B/A)*dA/dc + 0.5*sqrt(A/B)*dB/dc
        df_dc = 0.5 * np.sqrt(B / A) * dA_dc + 0.5 * np.sqrt(A / B) * dB_dc
        
        # Chain rule for theta, phi
        dtheta = np.array([ct*cp, ct*sp, -st])
        dphi = np.array([-st*sp, st*cp, 0.0])
        
        grad_theta = np.dot(df_dc, dtheta)
        grad_phi = np.dot(df_dc, dphi)
        
        return bound, np.array([grad_theta, grad_phi])

    def objective_with_grad(theta_phi):
        """Objective with gradient for SLSQP."""
        theta, phi = theta_phi
        val, grad = compute_bound_and_grad_from_spherical(theta, phi)
        return val, grad

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

    bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
    
    # Phase 1: Global optimization using Differential Evolution
    if time.time() - start_time < deadline - 60:
        try:
            de_result = differential_evolution(
                compute_bound_from_spherical, 
                bounds, 
                seed=42, 
                maxiter=150, 
                tol=1e-8,
                popsize=12,
                polish=False
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # Phase 2: Multi-start SLSQP with analytic gradients from diverse points
    if time.time() - start_time < deadline - 40:
        # Generate diverse starting points
        rng = np.random.RandomState(42)
        start_points = [(best_theta, best_phi)]
        
        # Add points from DE result if available
        if 'de_result' in dir() and hasattr(de_result, 'x'):
            start_points.append(tuple(de_result.x))
        
        # Add random points
        for _ in range(5):
            t = rng.uniform(0.1, np.pi - 0.1)
            p = rng.uniform(0.1, 2 * np.pi - 0.1)
            start_points.append((t, p))
        
        # Remove duplicates and run SLSQP from each
        for sp in start_points:
            if time.time() - start_time >= deadline - 30:
                break
            try:
                res = minimize(
                    objective_with_grad,
                    x0=np.array(sp),
                    jac=True,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_theta, best_phi = res.x
            except Exception:
                pass

    # Phase 3: Final high-precision polish with Nelder-Mead
    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 20000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Phase 4: One more SLSQP polish from the best point
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                objective_with_grad,
                x0=np.array([best_theta, best_phi]),
                jac=True,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
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
