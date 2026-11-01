"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def _hermite_h(n, x):
    """Probabilists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)

def _compute_c4_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for the test function
    g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    Uses analytical Fourier transform properties of Hermite functions.
    """
    # Hn(0) for even n
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # a12 from g(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute g(x) and its Fourier transform analytically
    # Using: F[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2} * sqrt(pi)
    # with convention F[f](k) = integral f(x) e^{-ikx} dx
    
    # For the unitary convention F[f](k) = (1/sqrt(2*pi)) integral f(x) e^{-ikx} dx:
    # F[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2} / sqrt(2)
    
    # Let's use a grid and compute numerically but efficiently
    x = np.linspace(-12, 12, 1000)
    dx = x[1] - x[0]
    
    # g(x)
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    exp_term = np.exp(-x**2 / 2)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * exp_term
    
    # L2 norm squared
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-15:
        return 1e10
    
    # Fourier transform using analytical property
    # F[g](k) = (a0 * (-i)^0 H0(k) + a4 * (-i)^4 H4(k) + a8 * (-i)^8 H8(k) + a12 * (-i)^12 H12(k)) * exp(-k^2/2) / sqrt(2)
    # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1 (all even powers of -i are 1 or -1)
    # (-i)^0 = 1
    # (-i)^4 = (-i)^4 = i^4 = 1
    # (-i)^8 = 1
    # (-i)^12 = 1
    # So all coefficients are +1 for even n!
    
    k = x  # Same grid for FT
    H0_k = _hermite_h(0, k)
    H4_k = _hermite_h(4, k)
    H8_k = _hermite_h(8, k)
    H12_k = _hermite_h(12, k)
    
    ft_g = (a0 * H0_k + a4 * H4_k + a8 * H8_k + a12 * H12_k) * np.exp(-k**2 / 2) / np.sqrt(2)
    
    ft_norm_sq = np.sum(ft_g**2) * dx
    
    if ft_norm_sq < 1e-15:
        return 1e10
    
    # Compute variances (second moments)
    var_x = np.sum(x**2 * g**2) * dx / norm_sq
    var_k = np.sum(k**2 * ft_g**2) * dx / ft_norm_sq
    
    if var_x <= 0 or var_k <= 0:
        return 1e10
    
    # The upper bound on C4: use 2*sigma_x*sigma_k as a proxy
    # Heisenberg says sigma_x * sigma_k >= 1/2, so 2*sigma_x*sigma_k >= 1
    c4_bound = 2.0 * np.sqrt(var_x) * np.sqrt(var_k)
    
    return c4_bound

def construct():
    """
    Return three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 190  # Safety margin
    
    # Initial guess
    initial_guess = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        try:
            return _compute_c4_bound(params[0], params[1], params[2])
        except Exception:
            return 1e10
    
    best_params = initial_guess.copy()
    best_score = objective(initial_guess)
    
    # Multiple starting points
    rng = np.random.default_rng(42)
    starting_points = [initial_guess]
    for _ in range(15):
        sp = rng.normal(loc=0, scale=0.3, size=3)
        starting_points.append(sp)
    
    for sp in starting_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Refine
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if result.fun < best_score:
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
