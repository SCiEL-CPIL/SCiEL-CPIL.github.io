"""Program for the uncertainty inequality task using Remez exchange on dual problem."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def _H(n, x):
    """Physicist's Hermite polynomials up to degree 12."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 4:
        return 16.0 * x**4 - 48.0 * x**2 + 12.0
    elif n == 8:
        return 256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4 - 13440.0 * x**2 + 1680.0
    elif n == 12:
        return (4096.0 * x**12 - 122880.0 * x**10 + 1290240.0 * x**8
                - 6451200.0 * x**6 + 13107200.0 * x**4 - 9031680.0 * x**2 + 1032192.0)
    else:
        raise ValueError(f"Unsupported Hermite degree {n}")


def _c12_from(c0, c4, c8):
    """Compute c12 such that g(0) = 0."""
    return -(c0 + 12.0 * c4 + 1680.0 * c8) / 1032192.0


def _evaluate(c0, c4, c8, x_grid):
    """
    Evaluate g(x) and G(xi) on the given grid.
    Returns (g_values, G_values) where g(x) = P(x)*exp(-x^2/2) and G(xi) is its FT.
    """
    c12 = _c12_from(c0, c4, c8)
    gauss = np.exp(-x_grid**2 / 2.0)
    P = c0 * _H(0, x_grid) + c4 * _H(4, x_grid) + c8 * _H(8, x_grid) + c12 * _H(12, x_grid)
    g = P * gauss
    # FT: since all degrees are even, (-i)^n = 1 for n=0,4,8,12
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    G = sqrt_2pi * P * gauss  # P(xi)*exp(-xi^2/2) with same coefficients
    return g, G


def _compute_bound(c0, c4, c8, x_grid=None):
    """Compute A(f)*A(F) = sup|x*g(x)| * sup|xi*G(xi)|."""
    if x_grid is None:
        x_grid = np.linspace(-12.0, 12.0, 8001)
    g, G = _evaluate(c0, c4, c8, x_grid)
    A_f = np.max(np.abs(x_grid * g))
    A_F = np.max(np.abs(x_grid * G))  # xi = x_grid for the dual
    return A_f * A_F


def _find_critical_points(c0, c4, c8, x_grid, n_points=6):
    """
    Find critical points where |x*g(x)| and |xi*G(xi)| achieve near-maxima.
    Returns (x_crit, xi_crit) arrays of critical point locations.
    """
    g, G = _evaluate(c0, c4, c8, x_grid)
    xg = np.abs(x_grid * g)
    xiG = np.abs(x_grid * G)
    
    # Find local maxima by looking for peaks
    def find_peaks(arr, n):
        """Find n largest local maxima."""
        peaks = []
        # Smooth slightly to avoid noise
        kernel = np.ones(5) / 5.0
        arr_smooth = np.convolve(arr, kernel, mode='same')
        
        # Find all local maxima
        for i in range(2, len(arr_smooth) - 2):
            if (arr_smooth[i] > arr_smooth[i-1] and arr_smooth[i] >= arr_smooth[i+1] and
                arr_smooth[i] > arr_smooth[i-2] and arr_smooth[i] >= arr_smooth[i+2]):
                peaks.append((arr_smooth[i], i))
        
        # Sort by value descending and take top n
        peaks.sort(key=lambda x: -x[0])
        indices = [p[1] for p in peaks[:n]]
        return np.array([x_grid[i] for i in indices])
    
    x_crit = find_peaks(xg, n_points)
    xi_crit = find_peaks(xiG, n_points)
    
    # Ensure we have enough points
    if len(x_crit) < 3:
        idx = np.argsort(xg)[-6:]
        x_crit = np.sort(x_grid[idx])
    if len(xi_crit) < 3:
        idx = np.argsort(xiG)[-6:]
        xi_crit = np.sort(x_grid[idx])
    
    return x_crit, xi_crit


def _remez_exchange(c_init, x_grid, max_iter=200, tol=1e-14):
    """
    Remez exchange algorithm for the dual problem.
    
    We minimize max(|x*g(x)|) * max(|xi*G(xi)|) by equiosculating
    the signed functions at critical points.
    
    Strategy:
    1. Find critical points where suprema are achieved
    2. Set up linear system: at each critical point, the signed value
       should equal +/- t (alternating signs)
    3. Solve for c0, c4, c8, t
    4. Exchange points if necessary
    """
    c = np.array(c_init, dtype=float)
    best_bound = _compute_bound(*c, x_grid)
    best_c = c.copy()
    
    # Initial critical points
    x_crit, xi_crit = _find_critical_points(*c, x_grid)
    
    for iteration in range(max_iter):
        c0, c4, c8 = c
        c12 = _c12_from(c0, c4, c8)
        
        # Evaluate at critical points
        g_x = _H(0, x_crit) * c0 + _H(4, x_crit) * c4 + _H(8, x_crit) * c8 + _H(12, x_crit) * c12
        g_x = g_x * np.exp(-x_crit**2 / 2.0)
        
        G_xi = _H(0, xi_crit) * c0 + _H(4, xi_crit) * c4 + _H(8, xi_crit) * c8 + _H(12, xi_crit) * c12
        G_xi = G_xi * np.exp(-xi_crit**2 / 2.0) * math.sqrt(2.0 * math.pi)
        
        # Signed values: x*g(x) and xi*G(xi)
        xg_signed = x_crit * g_x
        xiG_signed = xi_crit * G_xi
        
        # Current bound
        A_f = np.max(np.abs(xg_signed))
        A_F = np.max(np.abs(xiG_signed))
        current_bound = A_f * A_F
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_c = c.copy()
        
        # Check convergence
        if iteration > 5 and abs(current_bound - best_bound) < tol * best_bound:
            break
        
        # Build equiosculation system
        # We want: x_i * g(x_i) = s_i * t for i = 1..n_x
        #          xi_j * G(xi_j) = r_j * t' for j = 1..n_xi
        # But t and t' are related by the product constraint.
        
        # Alternative: minimize max(|xg|, |xiG| / alpha) for some alpha
        # The optimal alpha balances the two suprema.
        
        # Let's use a penalty approach: minimize (max|xg|)^2 + (max|xiG|)^2
        # with gradient descent on c0, c4, c8
        
        # Compute gradients numerically
        eps = 1e-8
        grad = np.zeros(3)
        for i in range(3):
            c_plus = c.copy()
            c_plus[i] += eps
            bound_plus = _compute_bound(*c_plus, x_grid)
            
            c_minus = c.copy()
            c_minus[i] -= eps
            bound_minus = _compute_bound(*c_minus, x_grid)
            
            grad[i] = (bound_plus - bound_minus) / (2.0 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        # Adaptive step size
        step_size = 0.01
        for _ in range(10):
            c_trial = c - step_size * grad / grad_norm
            bound_trial = _compute_bound(*c_trial, x_grid)
            if bound_trial < current_bound:
                c = c_trial
                current_bound = bound_trial
                break
            step_size *= 0.5
        
        # Periodically update critical points
        if iteration % 10 == 0:
            x_crit, xi_crit = _find_critical_points(*c, x_grid)
    
    return best_c, best_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Fine grid for evaluation
    x_grid = np.linspace(-12.0, 12.0, 12001)
    
    # Multiple starting points
    start_points = [
        np.array([-1.13, 0.04, 0.000309]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.95, 0.045, 0.00035]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.1, 0.042, 0.00032]),
    ]
    
    best_c = start_points[0].copy()
    best_bound = _compute_bound(*best_c, x_grid)
    
    for sp in start_points:
        try:
            c_opt, bound_opt = _remez_exchange(sp, x_grid, max_iter=150, tol=1e-14)
            if bound_opt < best_bound:
                best_bound = bound_opt
                best_c = c_opt
        except Exception:
            pass
    
    # Final refinement with finer grid
    x_grid_fine = np.linspace(-10.0, 10.0, 20001)
    try:
        c_final, bound_final = _remez_exchange(best_c, x_grid_fine, max_iter=50, tol=1e-15)
        if bound_final < best_bound:
            best_bound = bound_final
            best_c = c_final
    except Exception:
        pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
