"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
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


def _hermite_poly_scalar(n, x):
    """Compute physicist's Hermite polynomial Hn(x) for scalar x."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h0 = 1.0
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Effective basis functions incorporating the g(0)=0 constraint
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # c12 = -(c0 + 12*c4 + 1680*c8) / 40320
    # v_i(x) = (H_i(x) - coeff_i * H12(x) / 40320) * exp(-x^2/2)
    # where coeff_0=1, coeff_4=12, coeff_8=1680

    def v_func(i, x):
        """Compute basis function v_i(x) for scalar x."""
        if i == 0:
            return (_hermite_poly_scalar(0, x) - _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)
        elif i == 1:
            return (_hermite_poly_scalar(4, x) - 12.0 * _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)
        elif i == 2:
            return (_hermite_poly_scalar(8, x) - 1680.0 * _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)

    # Compute exact integrals using adaptive quadrature over the entire real line
    # M[i,j] = ∫ v_i(x) * v_j(x) dx
    # Mx[i,j] = ∫ x^2 * v_i(x) * v_j(x) dx
    # For momentum: use Parseval. 
    # The Fourier transform of H_n(x) * exp(-x^2/2) is related to H_n(xi) * exp(-xi^2/2) up to constants.
    # Specifically, if f(x) = H_n(x) e^{-x^2/2}, then F(xi) = sqrt(2*pi) * (-i)^n * H_n(xi) * e^{-xi^2/2}
    # Wait, let's check the normalization.
    # Standard result: F{H_n(x) e^{-x^2/2}}(xi) = sqrt(2*pi) * (-i)^n * H_n(xi) * e^{-xi^2/2}
    # So |F(xi)|^2 = 2*pi * H_n(xi)^2 * e^{-xi^2}
    # And ∫ xi^2 |F(xi)|^2 dxi = 2*pi * ∫ xi^2 H_n(xi)^2 e^{-xi^2} dxi
    #
    # But our basis functions are combinations. Let's use linearity.
    # g(x) = sum_i c_i v_i(x)
    # G(xi) = sum_i c_i V_i(xi) where V_i is FT of v_i.
    # Since v_i is a linear combination of H_k * exp(-x^2/2), its FT is a linear combination of H_k * exp(-xi^2/2).
    #
    # Let's define the "dual" basis in frequency space.
    # v_0(x) = (H0 - H12/40320) e^{-x^2/2}
    # V_0(xi) = sqrt(2*pi) * [1 * e^{-xi^2/2} - (1/40320) * (-i)^12 * H12(xi) * e^{-xi^2/2}]
    # (-i)^12 = (-i)^12 = ((-i)^4)^3 = 1^3 = 1.
    # So V_0(xi) = sqrt(2*pi) * (1 - H12(xi)/40320) * e^{-xi^2/2}
    #
    # Similarly:
    # v_1(x) = (H4 - 12*H12/40320) e^{-x^2/2}
    # V_1(xi) = sqrt(2*pi) * [(-i)^4 * H4(xi) - (12/40320) * 1 * H12(xi)] * e^{-xi^2/2}
    # (-i)^4 = 1.
    # V_1(xi) = sqrt(2*pi) * (H4(xi) - 12*H12(xi)/40320) * e^{-xi^2/2}
    #
    # v_2(x) = (H8 - 1680*H12/40320) e^{-x^2/2}
    # V_2(xi) = sqrt(2*pi) * [(-i)^8 * H8(xi) - (1680/40320) * 1 * H12(xi)] * e^{-xi^2/2}
    # (-i)^8 = 1.
    # V_2(xi) = sqrt(2*pi) * (H8(xi) - 1680*H12(xi)/40320) * e^{-xi^2/2}
    #
    # Notice that the frequency-space basis functions have the SAME functional form as the position-space ones,
    # just with the roles of the Hermite polynomials swapped in a specific way? No, they are identical!
    # V_i(xi) = sqrt(2*pi) * v_i(xi)  (up to the sign factors which are all +1 for even n)
    #
    # So ∫ |G(xi)|^2 dxi = 2*pi * ∫ |g(xi)|^2 dxi = 2*pi * norm2 (by Parseval, since ∫|g|^2 = ∫|G|^2/(2*pi))
    # Wait, Parseval: ∫ |g(x)|^2 dx = (1/2*pi) ∫ |G(xi)|^2 dxi
    # So ∫ |G(xi)|^2 dxi = 2*pi * ∫ |g(x)|^2 dx
    #
    # And ∫ xi^2 |G(xi)|^2 dxi = 2*pi * ∫ xi^2 |g(xi)|^2 dxi  (since the kernel is the same)
    #
    # So Mf[i,j] = 2*pi * Mx[i,j]
    #
    # This means delta_x * delta_xi = sqrt(x2_norm/norm2) * sqrt(2*pi * x2_norm / (2*pi * norm2))
    # = sqrt(x2_norm/norm2) * sqrt(x2_norm/norm2)
    # = x2_norm / norm2
    #
    # Wait, this implies the bound is just x2_norm / norm2. Let me double check.
    # delta_x = sqrt(∫x^2|g|^2 / ∫|g|^2)
    # delta_xi = sqrt(∫xi^2|G|^2 / ∫|G|^2)
    # ∫|G|^2 = 2*pi * ∫|g|^2
    # ∫xi^2|G|^2 = 2*pi * ∫xi^2|g(xi)|^2 dxi  (because G(xi) = sqrt(2*pi)*g(xi) for even Hermite)
    # So delta_xi = sqrt(2*pi * ∫xi^2|g(xi)|^2 / (2*pi * ∫|g|^2)) = sqrt(∫x^2|g|^2 / ∫|g|^2) = delta_x
    #
    # So the product is delta_x^2 = x2_norm / norm2.
    #
    # This is a huge simplification! The optimization reduces to minimizing x2_norm / norm2.

    # Precompute M and Mx using quad
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))

    quad_opts = {'epsabs': 1e-12, 'epsrel': 1e-12, 'limit': 200}

    for i in range(3):
        for j in range(i, 3):
            def integrand_M(x):
                return v_func(i, x) * v_func(j, x)

            def integrand_Mx(x):
                return x**2 * v_func(i, x) * v_func(j, x)

            Mij, _ = quad(integrand_M, -np.inf, np.inf, **quad_opts)
            Mxij, _ = quad(integrand_Mx, -np.inf, np.inf, **quad_opts)

            M[i, j] = Mij
            M[j, i] = Mij
            Mx[i, j] = Mxij
            Mx[j, i] = Mxij

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c using precomputed matrices."""
        c = np.array(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        
        if not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        # Bound is x2_norm / norm2
        return x2_norm / norm2

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
    theta0 = np.arccos(np.clip(c_seed_norm[2], -1.0, 1.0))
    phi0 = np.arctan2(c_seed_norm[1], c_seed_norm[0])
    if phi0 < 0:
        phi0 += 2 * np.pi

    # Phase 1: Global search with differential evolution
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
            de_result = differential_evolution(
                compute_bound_spherical,
                bounds,
                seed=42,
                maxiter=150,
                tol=1e-10,
                popsize=20,
                polish=False
            )
            if de_result.fun < best_bound and np.isfinite(de_result.fun):
                best_bound = de_result.fun
                theta_best, phi_best = de_result.x
                c0 = np.sin(theta_best) * np.cos(phi_best)
                c4 = np.sin(theta_best) * np.sin(phi_best)
                c8 = np.cos(theta_best)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(c_seed)
                best_coeffs = c_candidate * scale
        except Exception:
            pass

    # Phase 2: Multi-start Nelder-Mead from diverse points
    if time.time() - start_time < deadline - 20:
        rng = np.random.RandomState(42)
        
        # Generate diverse starting points on the sphere
        start_points = [(theta0, phi0)]
        
        # Golden angle spiral for uniform coverage
        golden_angle = np.pi * (3 - np.sqrt(5))
        n_spiral = 30
        for i in range(n_spiral):
            z = 1 - 2 * (i + 0.5) / n_spiral
            theta = np.arccos(np.clip(z, -1.0, 1.0))
            phi = golden_angle * i
            if phi < 0:
                phi += 2 * np.pi
            start_points.append((theta, phi))
        
        # Random points
        for _ in range(20):
            theta = np.arccos(2 * rng.random() - 1)
            phi = 2 * np.pi * rng.random()
            start_points.append((theta, phi))
        
        # Add the DE result if it improved
        if 'de_result' in dir() and de_result.fun < best_bound:
            start_points.append(tuple(de_result.x))
        
        best_spherical_bound = best_bound
        best_spherical_point = (theta0, phi0)
        
        for theta_init, phi_init in start_points:
            if time.time() - start_time > deadline - 10:
                break
            
            try:
                res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_init, phi_init]),
                    method='Nelder-Mead',
                    options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 5000, 'adaptive': True}
                )
                
                if res.fun < best_spherical_bound and np.isfinite(res.fun):
                    best_spherical_bound = res.fun
                    best_spherical_point = res.x
                    
                    c0 = np.sin(res.x[0]) * np.cos(res.x[1])
                    c4 = np.sin(res.x[0]) * np.sin(res.x[1])
                    c8 = np.cos(res.x[0])
                    c_candidate = np.array([c0, c4, c8])
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                    best_bound = best_spherical_bound
            except Exception:
                pass

    # Phase 3: Ultra-tight final polish
    if time.time() - start_time < deadline - 2:
        try:
            theta_best, phi_best = best_spherical_point
            
            # Multiple polish passes with decreasing tolerance
            for polish_iter in range(3):
                tol_factor = 10 ** (-14 - polish_iter)
                polish_res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_best, phi_best]),
                    method='Nelder-Mead',
                    options={'xatol': tol_factor, 'fatol': tol_factor / 100, 'maxiter': 20000, 'adaptive': True}
                )
                
                if polish_res.fun < best_bound and np.isfinite(polish_res.fun):
                    best_bound = polish_res.fun
                    theta_best, phi_best = polish_res.x
                    c0 = np.sin(theta_best) * np.cos(phi_best)
                    c4 = np.sin(theta_best) * np.sin(phi_best)
                    c8 = np.cos(theta_best)
                    c_candidate = np.array([c0, c4, c8])
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                else:
                    break
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
