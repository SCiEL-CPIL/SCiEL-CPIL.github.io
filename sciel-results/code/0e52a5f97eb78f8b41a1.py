"""Program for the uncertainty inequality task using Remez exchange algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        return hermite(n)(x)


def _hermite_deriv(n, x):
    """Compute derivative of nth physicist's Hermite polynomial."""
    if n == 0:
        return np.zeros_like(x)
    elif n == 4:
        return 64 * x**3 - 96 * x
    elif n == 8:
        return 2048 * x**7 - 21504 * x**5 + 53760 * x**3 - 26880 * x
    elif n == 12:
        return (49152 * x**11 - 1228800 * x**9 + 10321920 * x**7 
                - 38707200 * x**5 + 52428800 * x**3 - 18063360 * x)
    else:
        return hermite(n)(x).derivative()(x)


def _compute_bound_and_critical(c0, c4, c8, x_grid=None):
    """
    Compute A_f, A_F, and find critical points where |x*g| and |xi*G| achieve maxima.
    Returns (bound, A_f, A_F, x_crit_f, xi_crit_F)
    """
    if x_grid is None:
        x_grid = np.linspace(-12, 12, 20001)
    
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    gauss = np.exp(-x_grid**2 / 2)
    poly = (c0 * _hermite_poly(0, x_grid) + c4 * _hermite_poly(4, x_grid) + 
            c8 * _hermite_poly(8, x_grid) + c12 * _hermite_poly(12, x_grid))
    g = poly * gauss
    
    # A_f = sup |x * g(x)|
    xf = x_grid * g
    abs_xf = np.abs(xf)
    A_f = np.max(abs_xf)
    
    # Find critical points of |x*g(x)| - look for local maxima
    # Use sign changes in derivative to find peaks
    dx = x_grid[1] - x_grid[0]
    dxf = np.gradient(xf, dx)
    d2xf = np.gradient(dxf, dx)
    
    # Local maxima: dxf changes from + to -, and d2xf < 0
    sign_change = np.zeros_like(xf, dtype=bool)
    for i in range(1, len(xf)):
        if dxf[i-1] > 0 and dxf[i] <= 0:
            sign_change[i] = True
    
    x_crit_f = x_grid[sign_change]
    if len(x_crit_f) == 0:
        x_crit_f = x_grid[np.argmax(abs_xf)]
    
    # Similarly for Fourier side
    xi_grid = x_grid.copy()
    gauss_xi = np.exp(-xi_grid**2 / 2)
    poly_xi = (c0 * _hermite_poly(0, xi_grid) + c4 * _hermite_poly(4, xi_grid) + 
               c8 * _hermite_poly(8, xi_grid) + c12 * _hermite_poly(12, xi_grid))
    sqrt_2pi = math.sqrt(2 * math.pi)
    G = sqrt_2pi * poly_xi * gauss_xi
    
    xiG = xi_grid * G
    abs_xiG = np.abs(xiG)
    A_F = np.max(abs_xiG)
    
    dxiG = np.gradient(xiG, dx)
    sign_change_F = np.zeros_like(xiG, dtype=bool)
    for i in range(1, len(xiG)):
        if dxiG[i-1] > 0 and dxiG[i] <= 0:
            sign_change_F[i] = True
    
    xi_crit_F = xi_grid[sign_change_F]
    if len(xi_crit_F) == 0:
        xi_crit_F = xi_grid[np.argmax(abs_xiG)]
    
    bound = A_f * A_F
    return bound, A_f, A_F, x_crit_f, xi_crit_F


def _remez_step(c0, c4, c8, x_crit_f, xi_crit_F, max_iter=50, tol=1e-10):
    """
    Remez exchange: adjust coefficients to equiosculate error at critical points.
    We want |x*g(x)| to have equal magnitude at all x_crit_f points,
    and |xi*G(xi)| to have equal magnitude at all xi_crit_F points.
    
    Since g and G have the same polynomial structure (up to sqrt(2pi)),
    we can set up a system where the ratio |x*g(x)| / |xi*G(xi)| is balanced.
    
    Actually, the key insight: at the optimal solution, the "error" in the 
    uncertainty product should be equioscillating. We use the fact that
    both A_f and A_F depend linearly on the coefficients in their polynomial parts.
    
    Let P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12 where c12 = -(c0+12c4+1680c8)/1032192
    Then x*g(x) = x*P(x)*exp(-x^2/2)
    And xi*G(xi) = sqrt(2pi)*xi*P(xi)*exp(-xi^2/2)
    
    At critical points, we want to minimize max(|x*P(x)*exp(-x^2/2)|) * max(|xi*P(xi)*exp(-xi^2/2)|)
    
    Use iterative reweighting: at each step, compute weights at critical points
    and solve a weighted least squares to balance the values.
    """
    x_grid = np.linspace(-12, 12, 20001)
    
    for iteration in range(max_iter):
        c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
        
        # Evaluate at critical points
        x_f = x_crit_f
        xi_F = xi_crit_F
        
        # Compute values at critical points
        P_x = (c0 * _hermite_poly(0, x_f) + c4 * _hermite_poly(4, x_f) + 
               c8 * _hermite_poly(8, x_f) + c12 * _hermite_poly(12, x_f))
        vals_f = x_f * P_x * np.exp(-x_f**2 / 2)
        
        P_xi = (c0 * _hermite_poly(0, xi_F) + c4 * _hermite_poly(4, xi_F) + 
                c8 * _hermite_poly(8, xi_F) + c12 * _hermite_poly(12, xi_F))
        sqrt_2pi = math.sqrt(2 * math.pi)
        vals_F = sqrt_2pi * xi_F * P_xi * np.exp(-xi_F**2 / 2)
        
        # We want to find new coefficients such that |vals_f| are all equal
        # and |vals_F| are all equal, minimizing the product of the two suprema.
        
        # Build the linear system: for each critical point x_i,
        # x_i * exp(-x_i^2/2) * [H0(x_i), H4(x_i), H8(x_i), H12(x_i)] . c = target_f
        # Similarly for xi_j.
        
        # Since c12 is determined by c0,c4,c8, we have 3 free variables.
        # Let's express everything in terms of c0, c4, c8.
        
        # H12 contribution: c12 * H12(x) = -(c0+12c4+1680c8)/1032192 * H12(x)
        # So the effective coefficient vector for [c0, c4, c8] at point x is:
        # [H0(x) - H12(x)/1032192, H4(x) - 12*H12(x)/1032192, H8(x) - 1680*H12(x)/1032192]
        
        def effective_coeffs(x):
            H0 = _hermite_poly(0, x)
            H4 = _hermite_poly(4, x)
            H8 = _hermite_poly(8, x)
            H12 = _hermite_poly(12, x)
            e0 = H0 - H12 / 1032192.0
            e4 = H4 - 12 * H12 / 1032192.0
            e8 = H8 - 1680 * H12 / 1032192.0
            return e0, e4, e8
        
        # Build matrices for position and momentum sides
        e0_f, e4_f, e8_f = effective_coeffs(x_f)
        w_f = x_f * np.exp(-x_f**2 / 2)
        
        e0_F, e4_F, e8_F = effective_coeffs(xi_F)
        w_F = sqrt_2pi * xi_F * np.exp(-xi_F**2 / 2)
        
        # For position: vals_f[i] = w_f[i] * (e0_f[i]*c0 + e4_f[i]*c4 + e8_f[i]*c8)
        # For momentum: vals_F[j] = w_F[j] * (e0_F[j]*c0 + e4_F[j]*c4 + e8_F[j]*c8)
        
        # We want to minimize max_i |vals_f[i]| * max_j |vals_F[j]|
        # Equioscillation: make |vals_f[i]| = alpha for all i, |vals_F[j]| = beta for all j
        # with alpha*beta minimized.
        
        # Use iterative approach: fix alpha/beta ratio, solve for c that equioscillates.
        
        # Simpler: use subgradient method or alternating minimization.
        # Let's try: compute current vals, find which points are "active" (near max),
        # then adjust c to reduce the max while keeping others balanced.
        
        abs_vals_f = np.abs(vals_f)
        abs_vals_F = np.abs(vals_F)
        
        max_f = np.max(abs_vals_f)
        max_F = np.max(abs_vals_F)
        
        if max_f < 1e-15 or max_F < 1e-15:
            break
        
        # Weights: emphasize points that are near the maximum
        eps = 1e-6
        weights_f = np.exp(-(abs_vals_f - max_f) / (max_f * eps))
        weights_F = np.exp(-(abs_vals_F - max_F) / (max_F * eps))
        
        # Solve weighted least squares to find c that minimizes
        # sum_i weights_f[i] * (vals_f[i]^2 - target_f^2)^2 + similar for F
        # This is complex. Instead, use a simpler heuristic:
        # Perturb c in the direction that reduces the maximum.
        
        # Compute gradient of bound w.r.t. c0, c4, c8 numerically
        h = 1e-7
        c_current = np.array([c0, c4, c8])
        
        grad = np.zeros(3)
        for k in range(3):
            c_plus = c_current.copy()
            c_plus[k] += h
            b_plus, _, _, _, _ = _compute_bound_and_critical(*c_plus, x_grid)
            c_minus = c_current.copy()
            c_minus[k] -= h
            b_minus, _, _, _, _ = _compute_bound_and_critical(*c_minus, x_grid)
            grad[k] = (b_plus - b_minus) / (2 * h)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Take a step in negative gradient direction with adaptive step size
        step_size = 0.1
        c_new = c_current - step_size * grad / grad_norm
        
        # Check if new point is better
        b_new, _, _, xc_new, xic_new = _compute_bound_and_critical(*c_new, x_grid)
        b_old, _, _, _, _ = _compute_bound_and_critical(*c_current, x_grid)
        
        if b_new < b_old:
            c0, c4, c8 = c_new
            x_crit_f = xc_new
            xi_crit_F = xic_new
        else:
            # Halve step size and retry
            for _ in range(10):
                step_size *= 0.5
                c_new = c_current - step_size * grad / grad_norm
                b_new, _, _, xc_new, xic_new = _compute_bound_and_critical(*c_new, x_grid)
                if b_new < b_old:
                    c0, c4, c8 = c_new
                    x_crit_f = xc_new
                    xi_crit_F = xic_new
                    break
            else:
                break
    
    return c0, c4, c8


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    x_grid = np.linspace(-12, 12, 20001)
    
    # Initial guess
    c0, c4, c8 = -113.0/100, 1.0/25, 1.0/3240
    bound, A_f, A_F, x_crit_f, xi_crit_F = _compute_bound_and_critical(c0, c4, c8, x_grid)
    
    # Run Remez exchange iterations
    c0, c4, c8 = _remez_step(c0, c4, c8, x_crit_f, xi_crit_F, max_iter=100, tol=1e-10)
    
    # Final verification
    bound, _, _, _, _ = _compute_bound_and_critical(c0, c4, c8, x_grid)
    
    return np.array([c0, c4, c8])
# EVOLVE-BLOCK-END
