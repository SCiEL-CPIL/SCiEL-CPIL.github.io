"""Program for the uncertainty inequality task using Remez-like optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        # H_12(x) = 4096 x^12 - 122880 x^10 + 1290240 x^8 - 6451200 x^6 + 13107200 x^4 - 9031680 x^2 + 1032192
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        # Fallback using scipy
        return hermite(n)(x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F) for the given Hermite coefficients.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 1032192
    
    So c12 * 1032192 = -(c0 + 12*c4 + 1680*c8)
    c12 = -(c0 + 12*c4 + 1680*c8) / 1032192
    """
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    # We need to compute the bound on A(f) * A(F)
    # A(f) = sup_x |x f(x)| and A(F) = sup_x |xi F(xi)| where F is the Fourier transform
    # For this specific problem, the bound involves evaluating certain integrals or suprema
    
    # The standard approach: the bound C4 is determined by the worst-case ratio
    # We need to evaluate the supremum of |x g(x)| * |xi G(xi)| where G is FT of g
    
    # Let's compute this numerically
    # First, build g(x)
    x = np.linspace(-15, 15, 10001)
    gauss = np.exp(-x**2 / 2)
    g = (c0 * _hermite_poly(0, x) + c4 * _hermite_poly(4, x) + 
         c8 * _hermite_poly(8, x) + c12 * _hermite_poly(12, x)) * gauss
    
    # A(f) = sup |x g(x)|
    A_f = np.max(np.abs(x * g))
    
    # Now compute Fourier transform G(xi)
    # Using the fact that FT of H_n(x) exp(-x^2/2) relates to H_n(-xi) exp(-xi^2/2)
    # Actually, for physicist's Hermite: FT[H_n(x) e^{-x^2/2}] = (-i)^n H_n(xi) e^{-xi^2/2} * sqrt(2*pi)
    # Wait, let me be more careful. The standard FT convention: F(xi) = integral f(x) e^{-i x xi} dx
    # FT[e^{-x^2/2}] = sqrt(2*pi) * e^{-xi^2/2}
    # FT[H_n(x) e^{-x^2/2}] = (-i)^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
    
    xi = np.linspace(-15, 15, 10001)
    gauss_xi = np.exp(-xi**2 / 2)
    
    # FT of H0 * gauss = sqrt(2*pi) * gauss_xi
    # FT of H4 * gauss = (-i)^4 sqrt(2*pi) H4(xi) gauss_xi = sqrt(2*pi) H4(xi) gauss_xi
    # FT of H8 * gauss = (-i)^8 sqrt(2*pi) H8(xi) gauss_xi = sqrt(2*pi) H8(xi) gauss_xi
    # FT of H12 * gauss = (-i)^12 sqrt(2*pi) H12(xi) gauss_xi = sqrt(2*pi) H12(xi) gauss_xi
    
    sqrt_2pi = math.sqrt(2 * math.pi)
    G = sqrt_2pi * (c0 * _hermite_poly(0, xi) + c4 * _hermite_poly(4, xi) + 
                    c8 * _hermite_poly(8, xi) + c12 * _hermite_poly(12, xi)) * gauss_xi
    
    # A(F) = sup |xi G(xi)|
    A_F = np.max(np.abs(xi * G))
    
    return A_f * A_F


def _remez_optimize(c_init, max_iter=100, tol=1e-12):
    """
    Remez exchange algorithm for minimizing the upper bound C4.
    
    The key insight: the bound C4 = A_f * A_F where both are suprema.
    We can reformulate this as finding coefficients that make the product
    of the two suprema as small as possible, with the constraint that
    the error equiosculates at the critical points.
    
    Strategy: 
    1. For a given set of coefficients, find where |x*g(x)| and |xi*G(xi)| achieve their maxima
    2. These are the "critical points" where the bound is active
    3. Update coefficients to equiosculate (alternate signs) at these points
    """
    c = np.array(c_init, dtype=float)
    best_bound = _compute_bound(*c)
    best_c = c.copy()
    
    # Fine grid for finding critical points
    x_grid = np.linspace(-10, 10, 20001)
    xi_grid = np.linspace(-10, 10, 20001)
    
    for iteration in range(max_iter):
        c0, c4, c8 = c
        c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
        
        # Compute g(x) and G(xi)
        gauss_x = np.exp(-x_grid**2 / 2)
        gauss_xi = np.exp(-xi_grid**2 / 2)
        
        H0_x = np.ones_like(x_grid)
        H4_x = 16 * x_grid**4 - 48 * x_grid**2 + 12
        H8_x = 256 * x_grid**8 - 3584 * x_grid**6 + 13440 * x_grid**4 - 13440 * x_grid**2 + 1680
        H12_x = (4096 * x_grid**12 - 122880 * x_grid**10 + 1290240 * x_grid**8 
                 - 6451200 * x_grid**6 + 13107200 * x_grid**4 - 9031680 * x_grid**2 + 1032192)
        
        g_x = (c0 * H0_x + c4 * H4_x + c8 * H8_x + c12 * H12_x) * gauss_x
        
        H0_xi = np.ones_like(xi_grid)
        H4_xi = 16 * xi_grid**4 - 48 * xi_grid**2 + 12
        H8_xi = 256 * xi_grid**8 - 3584 * xi_grid**6 + 13440 * xi_grid**4 - 13440 * xi_grid**2 + 1680
        H12_xi = (4096 * xi_grid**12 - 122880 * xi_grid**10 + 1290240 * xi_grid**8 
                  - 6451200 * xi_grid**6 + 13107200 * xi_grid**4 - 9031680 * xi_grid**2 + 1032192)
        
        sqrt_2pi = math.sqrt(2 * math.pi)
        G_xi = sqrt_2pi * (c0 * H0_xi + c4 * H4_xi + c8 * H8_xi + c12 * H12_xi) * gauss_xi
        
        # Compute |x*g(x)| and |xi*G(xi)|
        xg = np.abs(x_grid * g_x)
        xiG = np.abs(xi_grid * G_xi)
        
        # Find the critical points (local maxima) of xg and xiG
        # Use a simple approach: find points where the value is close to the maximum
        max_xg = np.max(xg)
        max_xiG = np.max(xiG)
        
        # Find all points within 1% of the maximum
        x_crit = x_grid[np.abs(xg - max_xg) < 0.01 * max_xg]
        xi_crit = xi_grid[np.abs(xiG - max_xiG) < 0.01 * max_xiG]
        
        # If we have too many or too few, adjust threshold
        if len(x_crit) < 3:
            x_crit = x_grid[np.argsort(xg)[-10:]]
        if len(xi_crit) < 3:
            xi_crit = xi_grid[np.argsort(xiG)[-10:]]
        
        # Limit to top 7 points from each (alternating pattern expected)
        if len(x_crit) > 7:
            # Keep the most extreme points
            xg_at_crit = xg[np.isin(x_grid, x_crit, assume_unique=False)]
            x_crit = x_crit[np.argsort(xg_at_crit)[-7:]]
        
        if len(xi_crit) > 7:
            xiG_at_crit = xiG[np.isin(xi_grid, xi_crit, assume_unique=False)]
            xi_crit = xi_crit[np.argsort(xiG_at_crit)[-7:]]
        
        # Build the system of equations for equiosculation
        # We want the signed values at critical points to alternate in sign
        # and have equal magnitude (the optimal value t)
        
        # For x critical points: x_i * g(x_i) = s_i * t where s_i alternates
        # For xi critical points: xi_i * G(xi_i) = r_i * t where r_i alternates
        
        # This gives us a linear system in c0, c4, c8, t
        # But t is nonlinear... Let's use a different approach.
        
        # Alternative: minimize the maximum of |xg| and |xiG| separately
        # The product is minimized when both are balanced.
        
        # Let's try a direct optimization approach using the critical points
        # to guide the search.
        
        # Compute gradient-like information from the critical points
        # and do a local search
        
        # Find the current bound
        current_bound = np.max(xg) * np.max(xiG)
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_c = c.copy()
        
        # Check convergence
        if iteration > 0:
            prev_bound = _compute_bound(*best_c)
            if abs(prev_bound - current_bound) < tol * current_bound:
                break
        
        # Do a local search around the current point using the critical point information
        # Try small perturbations in the direction that reduces the bound
        step_sizes = [0.01, 0.001, 0.0001]
        directions = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
            np.array([1, 0, 1]),
            np.array([0, 1, 1]),
            np.array([1, 1, 1]),
            np.array([-1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, -1]),
            np.array([-1, -1, 0]),
            np.array([-1, 0, -1]),
            np.array([0, -1, -1]),
            np.array([-1, -1, -1]),
        ]
        
        improved = False
        for step in step_sizes:
            for d in directions:
                trial_c = c + step * d
                trial_bound = _compute_bound(*trial_c)
                if trial_bound < current_bound * (1 - 1e-6):
                    c = trial_c
                    current_bound = trial_bound
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try larger steps
            for step in [0.1, 0.05]:
                for d in directions:
                    trial_c = c + step * d
                    trial_bound = _compute_bound(*trial_c)
                    if trial_bound < current_bound * (1 - 1e-6):
                        c = trial_c
                        current_bound = trial_bound
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement, break
            break
    
    return best_c, best_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Start from the known good solution
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = _compute_bound(*best_c)
    
    # Run Remez optimization from multiple starting points
    start_points = [
        best_c,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.2, 0.05, 0.0005]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.1, 0.08, 0.001]),
        np.array([-1.05, 0.09, 0.0012]),
    ]
    
    for sp in start_points:
        try:
            c_opt, bound_opt = _remez_optimize(sp, max_iter=50, tol=1e-12)
            if bound_opt < best_bound:
                best_bound = bound_opt
                best_c = c_opt
        except Exception:
            pass
    
    # Final local refinement
    try:
        c_final, bound_final = _remez_optimize(best_c, max_iter=20, tol=1e-14)
        if bound_final < best_bound:
            best_bound = bound_final
            best_c = c_final
    except Exception:
        pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
