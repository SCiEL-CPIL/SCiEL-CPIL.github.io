import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 200  # 10s safety margin
    
    # Hermite polynomial values at 0
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # For g(x) = P(x) * exp(-x²/2), P(x) = sum c_k x^k
    # |g(x)|² = P(x)² * exp(-x²)
    # ∫ x^m P(x)² exp(-x²) dx can be computed using moments of the Gaussian.
    
    # For the Fourier transform with convention ĝ(ξ) = ∫ g(x) exp(-2πiξx) dx:
    # If g(x) = x^n exp(-x²/2), then ĝ(ξ) = i^n (d^n/dξ^n)[sqrt(2π) exp(-π²ξ²)]
    # Wait, let me be more careful.
    
    # FT of exp(-ax²) = sqrt(π/a) exp(-ξ²/(4a)) for FT convention ∫f(x)e^{-2πiξx}dx
    # For a = 1/2: FT of exp(-x²/2) = sqrt(2π) exp(-2π²ξ²)
    
    # Actually, let me use a simpler FT convention: ĝ(ξ) = ∫ g(x) e^{-iξx} dx
    # Then FT of exp(-x²/2) = sqrt(2π) exp(-ξ²/2)
    
    # For x^n exp(-x²/2): FT = i^n d^n/dξ^n [sqrt(2π) exp(-ξ²/2)]
    
    # This gets complicated for high n. Let me just use numerical integration but with a smarter approach.
    
    # Actually, let me reconsider. The problem says "upper bound on A(f)*A(Fourier(f))".
    # This is likely a specific constant that can be computed from the coefficients.
    
    # Let me try a different approach: use the fact that for these specific test functions,
    # the bound has a known analytical form.
    
    # Given the complexity, let me use a fast numerical approach with precomputed values.
    
    # Use a fine grid and vectorized operations
    x = np.linspace(-30, 30, 100001)
    dx = x[1] - x[0]
    
    def compute_bound_fast(coeffs):
        a, b, c = coeffs
        d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
        
        # P(x) coefficients: x^12, x^10, x^8, x^6, x^4, x^2
        p12 = d
        p10 = -60 * d
        p8 = c + 1260 * d
        p6 = -28 * c - 12600 * d
        p4 = b + 210 * c + 56700 * d
        p2 = -6 * b - 420 * c - 75600 * d
        
        # Compute P(x) on the grid
        x2 = x**2
        P_x = p12 * x2**6 + p10 * x2**5 + p8 * x2**4 + p6 * x2**3 + p4 * x2**2 + p2 * x2
        
        g_x = P_x * np.exp(-x2 / 2)
        g_abs2 = g_x**2
        
        norm_g = np.sum(g_abs2) * dx
        if norm_g < 1e-15:
            return 1e10
        
        x2_g = np.sum(x2 * g_abs2) * dx
        moment_x = x2_g / norm_g
        
        # For Fourier transform, use FFT
        G = np.fft.fft(g_x) * dx
        freqs = np.fft.fftfreq(len(x), d=dx)
        # Reorder to center
        G_shifted = np.fft.fftshift(G)
        freqs_shifted = np.fft.fftshift(freqs)
        
        G_abs2 = G_shifted.real**2 + G_shifted.imag**2
        norm_G = np.sum(G_abs2) * (2 * np.pi / len(x) / dx)  # Proper scaling
        
        # Actually, the scaling for FFT to approximate ∫f(x)e^{-2πiξx}dx is:
        # G_k ≈ dx * sum_j f(x_j) e^{-2πi x_j ξ_k}
        # So |G_k|² approximates |∫f(x)e^{-2πiξx}dx|² at ξ_k
        # The spacing in frequency is Δξ = 1/(N*dx)
        # So ∫|ĝ(ξ)|²dξ ≈ sum_k |G_k|² * Δξ = sum_k |G_k|² / (N*dx)
        
        norm_G = np.sum(G_abs2) / (len(x) * dx)
        
        if norm_G < 1e-15:
            return 1e10
        
        xi2_G = np.sum(freqs_shifted**2 * G_abs2) / (len(x) * dx)
        moment_xi = xi2_G / norm_G
        
        return moment_x * moment_xi
    
    # Initial guess
    x0 = np.array([-113/100, 1.0/25, 1.0/3240])
    
    best_bound = compute_bound_fast(x0)
    best_coeffs = x0.copy()
    
    # Optimize
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(compute_bound_fast, x0, method=method,
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
                x0 = result.x.copy()  # Use improved point for next method
        except Exception:
            pass
    
    # Random restarts
    np.random.seed(42)
    for i in range(100):
        if time.time() > deadline - 5:
            break
        x_try = np.random.uniform(-10, 10, size=3)
        try:
            bound = compute_bound_fast(x_try)
            if bound < best_bound:
                best_bound = bound
                best_coeffs = x_try.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
