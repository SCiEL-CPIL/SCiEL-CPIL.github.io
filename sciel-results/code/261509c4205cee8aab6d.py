import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    # Use recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    h_prev = _hermite_coeffs(n - 1)
    h_curr = _hermite_coeffs(n - 2) if n >= 2 else np.array([0.0])
    # h_curr represents H_{n-2}, h_prev represents H_{n-1}
    # H_n = 2x * H_{n-1} - 2(n-1) * H_{n-2}
    # 2x * H_{n-1}: shift coefficients up by 1, multiply by 2
    term1 = np.zeros(len(h_prev) + 1)
    term1[1:] = 2.0 * h_prev
    # 2(n-1) * H_{n-2}
    term2 = 2.0 * (n - 1) * h_curr
    # Pad term2 to same length
    max_len = max(len(term1), len(term2))
    term1 = np.pad(term1, (0, max_len - len(term1)))
    term2 = np.pad(term2, (0, max_len - len(term2)))
    return term1 - term2

def _eval_hermite_poly(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    # coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    result = np.zeros_like(x, dtype=float)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result

def _compute_g(c0, c4, c8, x):
    """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)"""
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    
    h0_coeffs = np.array([1.0])
    h4_coeffs = _hermite_coeffs(4)
    h8_coeffs = _hermite_coeffs(8)
    h12_coeffs = _hermite_coeffs(12)
    
    P = (c0 * _eval_hermite_poly(h0_coeffs, x) + 
         c4 * _eval_hermite_poly(h4_coeffs, x) + 
         c8 * _eval_hermite_poly(h8_coeffs, x) + 
         c12 * _eval_hermite_poly(h12_coeffs, x))
    
    return P * np.exp(-x**2 / 2.0)

def _compute_uncertainty_bound(c0, c4, c8):
    """Compute the uncertainty bound A(f)*A(F(f)) for f(x) = g(x)."""
    # Use numerical integration with sufficient points
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    f = _compute_g(c0, c4, c8, x)
    f_abs_sq = np.abs(f)**2
    
    # ||f||^2
    norm_sq = np.sum(f_abs_sq) * dx
    
    if norm_sq < 1e-15:
        return np.inf
    
    # ||x f||^2
    x2_f_abs_sq = x**2 * f_abs_sq
    x_norm_sq = np.sum(x2_f_abs_sq) * dx
    
    # For the Fourier side, use the fact that F[H_k e^{-x^2/2}] = i^k H_k e^{-x^2/2}
    # So ||xi F(f)||^2 = ||x f||^2 by symmetry? No, that's not right.
    # Actually, for the standard Fourier transform, the momentum variance equals 
    # the variance of the derivative or similar.
    # Let's compute numerically using FFT.
    
    # Use a grid suitable for FFT
    N = 32768
    L = 30.0  # domain [-15, 15]
    x_fft = np.linspace(-L, L, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    f_fft = _compute_g(c0, c4, c8, x_fft)
    
    # Compute Fourier transform
    F_f = np.fft.fftshift(np.fft.fft(f_fft)) * dx_fft
    xi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx_fft))
    
    F_f_abs_sq = np.abs(F_f)**2
    xi2_F_f_abs_sq = xi**2 * F_f_abs_sq
    
    xi_norm_sq = np.sum(xi2_F_f_abs_sq) * dx_fft
    
    # Uncertainty product: sqrt(||x f||^2 / ||f||^2) * sqrt(||xi F(f)||^2 / ||f||^2)
    # Or just ||x f|| * ||xi F(f)|| / ||f||^2
    pos_var = x_norm_sq / norm_sq
    mom_var = xi_norm_sq / norm_sq
    
    return np.sqrt(pos_var * mom_var)

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s kill
    
    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    seed_bound = _compute_uncertainty_bound(c0_init, c4_init, c8_init)
    
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    best_bound = seed_bound
    
    # Try multiple starting points
    starts = [
        [c0_init, c4_init, c8_init],
        [0.0, 0.04, 0.0003],
        [-1.0, 0.05, 0.0004],
        [-1.2, 0.03, 0.0002],
        [-0.8, 0.06, 0.0005],
    ]
    
    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            def objective(vars):
                c0, c4, c8 = vars
                return _compute_uncertainty_bound(c0, c4, c8)
            
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            def objective(vars):
                c0, c4, c8 = vars
                return _compute_uncertainty_bound(c0, c4, c8)
            
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
