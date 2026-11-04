import numpy as np
import time
from scipy.optimize import minimize

def _hermite(n, x):
    """Compute the n-th Hermite polynomial using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_n_minus_2 = np.ones_like(x)
    H_n_minus_1 = 2 * x
    for k in range(2, n + 1):
        H_n = 2 * x * H_n_minus_1 - 2 * (k - 1) * H_n_minus_2
        H_n_minus_2 = H_n_minus_1
        H_n_minus_1 = H_n
    return H_n_minus_1

def _compute_c12(c0, c4, c8):
    """Calculate c12 such that g(0) = 0."""
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def _objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # Define the polynomial part P(x)
    # We evaluate on a grid to compute norms numerically
    # Grid needs to be large enough to capture tails of Gaussian
    x = np.linspace(-15, 15, 10000)
    dx = x[1] - x[0]
    
    # Hermite polynomials
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 2.0)
    
    # Check if g is effectively zero
    norm_g_sq = np.sum(g**2) * dx
    if norm_g_sq < 1e-12:
        return 1e10 # Penalty for zero function
        
    # Fourier Transform
    # Using FFT for efficiency
    # We need to handle the shift and scaling for FFT
    # x is from -15 to 15. 
    # To use FFT, we assume x is uniformly spaced.
    # FFT computes DFT. We need to map to continuous FT.
    # FT[g](xi) = integral g(x) e^{-2pi i x xi} dx (if using 2pi convention)
    # Or FT[g](xi) = integral g(x) e^{-i x xi} dx (if using 1 convention)
    # Standard math convention in uncertainty principle: 
    # f(x) = integral hat(f)(xi) e^{i x xi} dxi
    # hat(f)(xi) = 1/sqrt(2pi) integral f(x) e^{-i x xi} dx
    # Let's use scipy.fft.fft
    
    # Zero pad to avoid aliasing and get finer frequency resolution
    N = len(x)
    N_fft = 2**16 # 65536
    g_padded = np.zeros(N_fft, dtype=np.complex128)
    # Place g in the middle? No, x starts at -15.
    # FFT assumes input is periodic and starts at index 0 corresponding to x=0? 
    # No, standard FFT input array corresponds to x_k = k * dx.
    # Our x_k = -15 + k * dx.
    # We can shift the array so that x=0 is at index 0? 
    # Easier: Use the property that if x is shifted, FT gets a phase factor.
    # Or just use scipy.fft.fftshift and ifftshift carefully.
    
    # Let's stick to a simpler numerical integration for FT if speed allows, 
    # but FFT is faster. 
    # Let's use the direct FFT approach with proper scaling.
    
    # Shift g so that the center of the domain is at index 0?
    # Actually, let's just use the definition:
    # xi_grid = fftfreq(N_fft, d=dx) * 2 * np.pi  (if using angular frequency xi)
    # But standard FFT uses e^{-2pi i k n / N}.
    # Let's use scipy.fft.fft
    
    g_fft = np.fft.fft(g_padded) * dx
    # This computes sum g_j e^{-2pi i j k / N} dx.
    # This is an approximation of integral g(x) e^{-2pi i x xi} dx 
    # if x_j = j * dx. But our x_j = -15 + j * dx.
    # So g_j = g(-15 + j dx).
    # The sum is sum g(-15+j dx) e^{-2pi i (j dx) (k dx / (N dx))} ... 
    # This is messy.
    
    # Alternative: Use numerical integration for FT on a smaller grid or use the fact that
    # we only need the L2 norm of xi * hat(g).
    # Parseval's theorem: ||xi hat(g)||_2^2 = ||g'||_2^2 / (2pi)^2 ?
    # If hat(f)(xi) = 1/sqrt(2pi) int f(x) e^{-ixi} dx, then f(x) = 1/sqrt(2pi) int hat(f)(xi) e^{ixi} dxi.
    # Then d/dx f(x) = 1/sqrt(2pi) int (i xi) hat(f)(xi) e^{ixi} dxi.
    # So ||f'||_2 = ||xi hat(f)||_2.
    # And ||x f||_2 = ||hat(f)'||_2.
    # So the uncertainty product U = ||x g||_2 ||g'||_2 / ||g||_2^2.
    # This avoids computing the FT explicitly! We just need derivatives.
    
    # Compute g'(x) numerically
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = P'(x) e^{-x^2/2} + P(x) (-x) e^{-x^2/2} = (P'(x) - x P(x)) e^{-x^2/2}
    
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P'(x) = c4 H4' + c8 H8' + c12 H12' (H0' = 0)
    
    # Derivatives of Hermite polynomials:
    # H_n'(x) = 2n H_{n-1}(x)
    
    H4_prime = 2 * 4 * _hermite(3, x)
    H8_prime = 2 * 8 * _hermite(7, x)
    H12_prime = 2 * 12 * _hermite(11, x)
    
    P_prime = c4 * H4_prime + c8 * H8_prime + c12 * H12_prime
    
    g_prime = (P_prime - x * P) * np.exp(-x**2 / 2.0)
    
    # Compute norms
    # ||g||_2^2 = integral g^2
    # ||x g||_2^2 = integral x^2 g^2
    # ||g'||_2^2 = integral g'^2
    
    norm_g_sq = np.sum(g**2) * dx
    norm_xg_sq = np.sum((x * g)**2) * dx
    norm_gprime_sq = np.sum(g_prime**2) * dx
    
    if norm_g_sq < 1e-12:
        return 1e10
        
    # Uncertainty Product
    U = np.sqrt(norm_xg_sq * norm_gprime_sq) / norm_g_sq
    
    return U

def construct():
    start_time = time.time()
    timeout = 180.0
    
    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # We will perform a few local optimizations from different starting points
    # to avoid local minima.
    
    best_coeffs = x0
    best_score = _objective(x0)
    
    # Define a few random perturbations around the initial guess
    # and some other heuristic starting points
    starts = [x0]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbation = rng.normal(0, 0.1, size=3)
        starts.append(x0 + perturbation)
    
    # Try to optimize from each start
    for start in starts:
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = minimize(_objective, start, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs

# Ensure the function is named construct and returns the array
# The evaluator calls construct()
