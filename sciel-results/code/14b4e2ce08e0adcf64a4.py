"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hermite_physicists(n, x):
    """Compute physicists' Hermite polynomial H_n(x) recursively."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2 * x
        for k in range(1, n):
            H2 = 2 * x * H1 - 2 * k * H0
            H0, H1 = H1, H2
        return H1

def compute_bound(a0, a4, a8, n_points=20000, x_range=15.0):
    """Compute the upper bound on C4 for given coefficients."""
    # Hermite polynomials
    x = np.linspace(-x_range, x_range, n_points)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 1280*x**6 + 2880*x**4 - 1920*x**2 + 168
    H12 = (4096*x**12 - 46080*x**10 + 230400*x**8 - 483840*x**6 + 403200*x**4 - 120960*x**2 + 10080)
    
    # Constraint g(0) = 0: a0*1 + a4*12 + a8*168 + a12*10080 = 0
    a12 = -(a0 + 12*a4 + 168*a8) / 10080.0
    
    # Gaussian weight
    gauss = np.exp(-0.5 * x**2)
    
    g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    # L1 norm of g
    l1_g = np.sum(np.abs(g)) * dx
    
    # Fourier transform: for g(x) = p(x) * exp(-x^2/2), 
    # the FT is (1/sqrt(2*pi)) * integral p(x) exp(-x^2/2) exp(-ixxi) dx
    # We can compute this numerically
    xi = np.linspace(-x_range, x_range, n_points)
    dxi = xi[1] - xi[0]
    
    # Use FFT for efficiency
    # g(x) is defined on x, we want its FT
    # FT convention: hat{g}(xi) = integral g(x) exp(-i*x*xi) dx
    # Using FFT: if x_k = -x_max + k*dx, then FT at xi_m can be approximated
    # Let's use direct numerical integration via FFT
    
    # Shift to use FFT properly
    # x goes from -x_range to x_range
    # FT at xi: sum g(x_k) exp(-i x_k xi) dx
    # We want xi values that are multiples of 2*pi/(N*dx)
    # N = n_points, dx = 2*x_range/N
    # Frequency spacing = 2*pi/(N*dx) = 2*pi/(2*x_range) = pi/x_range
    
    N = n_points
    dt = 2.0 * x_range / N  # this is dx
    # FFT frequency spacing: 2*pi/(N*dt) = 2*pi/(2*x_range) = pi/x_range
    
    # Use np.fft.fft
    # g is sampled at x_k = -x_range + k*dx for k=0,...,N-1
    # The FFT gives sum_k g_k exp(-2*pi*i*k*m/N)
    # We need sum_k g_k exp(-i*x_k*xi_m)
    # x_k = -x_range + k*dx = -x_range + k*2*x_range/N
    # exp(-i*x_k*xi) = exp(-i*(-x_range + k*2*x_range/N)*xi) = exp(i*x_range*xi) * exp(-i*k*2*x_range*xi/N)
    # For xi_m = m*pi/x_range: exp(-i*k*2*x_range*(m*pi/x_range)/N) = exp(-i*k*2*pi*m/N)
    # So FFT of g gives sum_k g_k exp(-2*pi*i*k*m/N), and we multiply by exp(i*x_range*xi_m)
    
    G_fft = np.fft.fft(g)
    # Frequency values
    xi_vals = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    # But we need the phase correction: exp(i * x_range * xi_m)
    # Actually, let me be more careful.
    # x_k = -x_range + k*dx
    # We want hat{g}(xi) = sum_k g(x_k) exp(-i*x_k*xi) * dx
    # = dx * exp(i*x_range*xi) * sum_k g(x_k) exp(-i*k*dx*xi)
    # For xi = m * 2*pi/(N*dx): exp(-i*k*dx*xi) = exp(-i*k*2*pi*m/N)
    # So sum_k g(x_k) exp(-i*k*2*pi*m/N) = FFT[g]_m
    
    # The FFT frequencies are: 0, 1/dt, 2/dt, ..., (N/2-1)/dt, -N/2/dt, ..., -1/dt
    # where dt = dx
    # So xi_m = m * 2*pi/(N*dx) for m = 0, ..., N/2, and negative for the rest
    
    # Phase factor: exp(i * x_range * xi_m)
    phase = np.exp(1j * x_range * xi_vals)
    hat_g = dx * phase * G_fft
    
    # Reorder to go from -xi_max to +xi_max
    hat_g = np.fft.fftshift(hat_g)
    xi_vals = np.fft.fftshift(xi_vals)
    
    l1_hat_g = np.sum(np.abs(hat_g)) * dxi
    
    # The bound C4 is typically l1_g * l1_hat_g / (l2_g^2 * l2_hat_g^2) or similar
    # Let's compute the standard uncertainty product bound
    # A(f) = ||f||_1, A(hat{f}) = ||hat{f}||_1
    # The constant C4 in the inequality A(f)*A(hat{f}) >= C4 * ||f||_2^2 * ||hat{f}||_2^2
    # For a test function g, the optimal C4 is bounded by:
    # C4 <= (||g||_1 * ||hat{g}||_1) / (||g||_2^2 * ||hat{g}||_2^2)
    
    l2_g_sq = np.sum(g**2) * dx
    l2_hat_g_sq = np.sum(hat_g**2) * dxi
    
    if l2_g_sq <= 0 or l2_hat_g_sq <= 0:
        return 1e10
    
    C4_bound = (l1_g * l1_hat_g) / (l2_g_sq * l2_hat_g_sq)
    
    return C4_bound

def objective(coeffs):
    """Objective function to minimize."""
    a0, a4, a8 = coeffs
    try:
        val = compute_bound(a0, a4, a8, n_points=5000, x_range=10.0)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Seed value
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2], n_points=5000, x_range=10.0)
    
    # Try multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.00035]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.13, 0.04, 0.000308642]) + 0.01 * np.random.randn(3),
        np.array([-1.13, 0.04, 0.000308642]) - 0.01 * np.random.randn(3),
    ]
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Refine with more points
    if time.time() < deadline - 30:
        final_val = compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2], n_points=10000, x_range=12.0)
        if final_val < best_val:
            best_val = final_val
    
    return best_coeffs
# EVOLVE-BLOCK-END
