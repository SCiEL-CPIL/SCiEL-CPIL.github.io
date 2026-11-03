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
    deadline = 175  # seconds, keep margin for evaluator kill at 210s

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
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices for fast evaluation
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
        """Compute bound for a specific coefficient vector c using precomputed matrices."""
        c = np.array(c, dtype=float)
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

    # Structural change: Use spherical coordinates to remove scale ambiguity
    # The objective is scale-invariant: f(c) = f(αc) for any α > 0
    # So we can fix ||c|| = 1 and optimize over the 2D sphere S^2
    # This reduces the search space and avoids scale-related issues
    
    def compute_bound_spherical(theta, phi):
        """Compute bound using spherical coordinates on the unit sphere."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return compute_bound_from_coeffs(c)

    # Initial best guess from seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = compute_bound_from_coeffs(c_seed)
    best_coeffs = c_seed.copy()
    
    # Normalize seed to unit sphere for spherical search
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    
    # Convert to spherical coordinates
    theta0 = np.arccos(c_seed_norm[2])  # arccos(z)
    phi0 = np.arctan2(c_seed_norm[1], c_seed_norm[0])  # arctan2(y, x)
    if phi0 < 0:
        phi0 += 2 * np.pi

    # Multi-start optimization on the sphere using multiple initial points
    # This is a structural change: instead of one global search, we do
    # many local searches from diverse starting points on the sphere
    
    if time.time() - start_time < deadline - 30:
        # Generate diverse starting points on the sphere
        n_starts = 50
        rng = np.random.RandomState(42)
        
        # Include the seed point and random points
        start_points = [(theta0, phi0)]
        for _ in range(n_starts - 1):
            theta = np.arccos(2 * rng.random() - 1)
            phi = 2 * np.pi * rng.random()
            start_points.append((theta, phi))
        
        # Also include some structured points (vertices of icosahedron approximations)
        # Golden angle spiral
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(20):
            z = 1 - 2 * (i + 0.5) / 20
            r = np.sqrt(1 - z**2)
            theta = np.arccos(z)
            phi = golden_angle * i
            if phi < 0:
                phi += 2 * np.pi
            start_points.append((theta, phi))
        
        best_spherical_bound = np.inf
        best_spherical_point = (theta0, phi0)
        
        for theta_init, phi_init in start_points:
            if time.time() - start_time > deadline - 10:
                break
            
            try:
                res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_init, phi_init]),
                    method='Nelder-Mead',
                    options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'adaptive': True}
                )
                
                if res.fun < best_spherical_bound and np.isfinite(res.fun):
                    best_spherical_bound = res.fun
                    best_spherical_point = res.x
                    
                    # Convert back to Cartesian
                    c0 = np.sin(res.x[0]) * np.cos(res.x[1])
                    c4 = np.sin(res.x[0]) * np.sin(res.x[1])
                    c8 = np.cos(res.x[0])
                    c_candidate = np.array([c0, c4, c8])
                    
                    # Scale to match original magnitude for comparison
                    # The objective is scale-invariant, so any scale works
                    # But we need to return a valid coefficient vector
                    # Use the seed magnitude as a reference
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                    
                    if best_spherical_bound < best_bound:
                        best_bound = best_spherical_bound
            except Exception:
                pass
    
    # Final high-precision polish: use the best point found so far and run
    # a very tight local optimization to squeeze out the last bits of precision.
    if time.time() - start_time < deadline - 2:
        try:
            # Convert best_coeffs back to spherical coordinates for the polish
            c_best = best_coeffs / np.linalg.norm(best_coeffs)
            theta_best = np.arccos(np.clip(c_best[2], -1.0, 1.0))
            phi_best = np.arctan2(c_best[1], c_best[0])
            if phi_best < 0:
                phi_best += 2 * np.pi
            
            # Run a very tight Nelder-Mead polish
            polish_res = minimize(
                compute_bound_spherical,
                x0=np.array([theta_best, phi_best]),
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-16, 'maxiter': 10000, 'adaptive': True}
            )
            
            if polish_res.fun < best_bound and np.isfinite(polish_res.fun):
                best_bound = polish_res.fun
                theta, phi = polish_res.x
                c0 = np.sin(theta) * np.cos(phi)
                c4 = np.sin(theta) * np.sin(phi)
                c8 = np.cos(theta)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(c_seed)
                best_coeffs = c_candidate * scale
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
