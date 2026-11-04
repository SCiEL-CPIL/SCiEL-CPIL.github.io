import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def construct():
    t_start = time.time()
    deadline = 195.0  # 15s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 3991680.0

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    def compute_bound(a0, a4, a8, x_grid=None):
        """Compute the uncertainty product bound for given coefficients."""
        a12 = compute_a12(a0, a4, a8)
        
        # Coefficients for powers x^0, x^2, x^4, x^6, x^8, x^10, x^12
        c0 = a0 * 1 + a4 * 12 + a8 * 1680 + a12 * 3991680
        c2 = a4 * (-48) + a8 * (-13440) + a12 * (-12355200)
        c4 = a4 * 16 + a8 * 11520 + a12 * 13271040
        c6 = a8 * (-3072) + a12 * (-5806080)
        c8 = a8 * 256 + a12 * 1032192
        c10 = a12 * (-98304)
        c12 = a12 * 4096
        
        if x_grid is None:
            x = np.linspace(-8, 8, 20001)
            x2 = x**2
            x4 = x2**2
            x6 = x4 * x2
            x8 = x4**2
            x10 = x8 * x2
            x12 = x8 * x4
            dx = x[1] - x[0]
        else:
            x = x_grid
            x2 = x**2
            x4 = x2**2
            x6 = x4 * x2
            x8 = x4**2
            x10 = x8 * x2
            x12 = x8 * x4
            dx = x[1] - x[0]
        
        P = c0 + c2*x2 + c4*x4 + c6*x6 + c8*x8 + c10*x10 + c12*x12
        g = P * np.exp(-np.pi * x2)
        
        norm_g = np.max(np.abs(g))
        
        N = len(x)
        G = np.fft.fftshift(np.fft.fft(g)) * dx
        norm_g_hat = np.max(np.abs(G))
        
        return norm_g * norm_g_hat

    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_score = compute_bound(*seed)
    best_params = seed.copy()
    best_score = seed_score

    # Phase 1: Differential Evolution in spherical coordinates (2D)
    # Remove scale ambiguity by parameterizing direction only
    def spherical_to_cartesian(theta, phi):
        r = 1.0  # Unit sphere
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    def objective_spherical(params):
        theta, phi = params
        # Map from sphere to reasonable coefficient range
        # Scale factor to cover the search space
        scale = 2.0
        a0, a4, a8 = spherical_to_cartesian(theta, phi)
        a0 *= scale
        a4 *= scale * 0.1  # a4 tends to be smaller
        a8 *= scale * 0.01  # a8 tends to be even smaller
        return compute_bound(a0, a4, a8)

    # Run differential evolution
    try:
        result_de = differential_evolution(
            objective_spherical,
            bounds=[(-np.pi, np.pi), (0, np.pi)],
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result_de.fun < best_score:
            theta, phi = result_de.x
            a0, a4, a8 = spherical_to_cartesian(theta, phi)
            a0 *= 2.0
            a4 *= 2.0 * 0.1
            a8 *= 2.0 * 0.01
            best_params = np.array([a0, a4, a8])
            best_score = result_de.fun
    except Exception as e:
        pass

    # Phase 2: Local polish with Nelder-Mead from best point
    if time.time() < deadline - 30:
        try:
            result_nm = minimize(
                lambda p: compute_bound(*p),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result_nm.fun < best_score:
                best_params = result_nm.x.copy()
                best_score = result_nm.fun
        except Exception as e:
            pass

    # Phase 3: Try multiple random restarts with local optimization
    if time.time() < deadline - 10:
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            try:
                # Random starting point in reasonable range
                rand_params = np.array([
                    np.random.uniform(-2.0, 0.0),
                    np.random.uniform(-0.1, 0.2),
                    np.random.uniform(-0.01, 0.02)
                ])
                
                result_local = minimize(
                    lambda p: compute_bound(*p),
                    rand_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                if result_local.fun < best_score:
                    best_params = result_local.x.copy()
                    best_score = result_local.fun
            except Exception as e:
                continue

    # Final verification
    final_score = compute_bound(*best_params)
    if final_score < best_score:
        best_score = final_score

    return best_params
