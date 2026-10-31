import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

DEADLINE = time.time() + 200  # 200s deadline, 10s margin

def _compute_uncertainty(a0, a4, a8):
    """Compute the uncertainty product for given coefficients."""
    if time.time() > DEADLINE:
        return None
    
    # H_n(0) for even n
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 168.0
    H12_0 = -4320.0
    
    # Constraint: g(0) = 0
    # a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def g(x):
        return (a0 * 1 + a4 * (x**4 - 6*x**2 + 3) + 
                a8 * (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105) +
                a12 * (x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945)) * np.exp(-x**2 / 2)
    
    # Compute ||g||_2^2
    norm2, _ = quad(lambda x: g(x)**2, -np.inf, np.inf, limit=200)
    if norm2 <= 0:
        return np.inf
    
    # Compute ∫ x^2 |g|^2
    x2_norm, _ = quad(lambda x: x**2 * g(x)**2, -np.inf, np.inf, limit=200)
    
    # Variance in x: Δx^2 = x2_norm / norm2
    dx2 = x2_norm / norm2
    
    # Fourier transform of g. For g(x) = P(x) exp(-x^2/2), 
    # the Fourier transform is related to the coefficients.
    # Using the property that FT of H_n(x) exp(-x^2/2) = i^n H_n(x) exp(-x^2/2) (with appropriate convention)
    # But let's compute numerically to be safe.
    
    # Actually, let me use the fact that for the standard Fourier transform
    # f̂(ξ) = ∫ f(x) exp(-2πiξx) dx
    # The FT of exp(-x^2/2) is sqrt(2π) exp(-2π^2 ξ^2)
    # This gets complicated. Let me use a different approach.
    
    # Alternative: use the property that Δx * Δp = sqrt(∫x^2|f|^2/∫|f|^2) * sqrt(∫ξ^2|f̂|^2/∫|f̂|^2)
    # And by Plancherel, ∫|f̂|^2 = ∫|f|^2 (with proper normalization)
    
    # Let me compute the Fourier transform numerically using FFT or quad
    # Actually, for a function that decays like exp(-x^2/2), I can compute the FT numerically
    
    # Better approach: use the analytic FT of Hermite-Gaussians
    # With convention f̂(ξ) = (1/sqrt(2π)) ∫ f(x) exp(-ixξ) dx
    # FT of H_n(x) exp(-x^2/2) = i^n H_n(x) exp(-x^2/2)
    
    # So ĝ(ξ) = (1/sqrt(2π)) * (a0*1 + a4*i^4*H4(ξ) + a8*i^8*H8(ξ) + a12*i^12*H12(ξ)) * exp(-ξ^2/2)
    # = (1/sqrt(2π)) * (a0*1 + a4*1*H4(ξ) + a8*1*H8(ξ) + a12*1*H12(ξ)) * exp(-ξ^2/2)
    # since i^4 = i^8 = i^12 = 1
    
    # So ĝ has the same form as g but with 1/sqrt(2π) factor!
    
    # Therefore ∫ ξ^2 |ĝ|^2 dξ = (1/(2π)) * ∫ ξ^2 |g(ξ)|^2 dξ = (1/(2π)) * x2_norm
    
    # And ∫ |ĝ|^2 dξ = (1/(2π)) * ∫ |g|^2 = (1/(2π)) * norm2
    
    # So Δp^2 = ∫ξ^2|ĝ|^2 / ∫|ĝ|^2 = x2_norm / norm2 = Δx^2
    
    # Wait, that means Δx = Δp always? That can't be right for all functions.
    
    # Let me reconsider the Fourier transform convention and the FT of Hermite-Gaussians.
    
    # With f̂(ξ) = (1/sqrt(2π)) ∫ f(x) e^{-ixξ} dx
    # The FT of exp(-x^2/2) is exp(-ξ^2/2)
    # The FT of x*exp(-x^2/2) = i*exp(-ξ^2/2) (up to sign convention)
    # More generally, FT of H_n(x) exp(-x^2/2) = i^n H_n(ξ) exp(-ξ^2/2)
    
    # So for our g(x) = P(x) exp(-x^2/2) where P is even polynomial:
    # ĝ(ξ) = (1/sqrt(2π)) * P(ξ) * exp(-ξ^2/2)  (since i^{even} = 1)
    
    # So |ĝ(ξ)|^2 = (1/(2π)) * |P(ξ)|^2 * exp(-ξ^2)
    # And |g(x)|^2 = |P(x)|^2 * exp(-x^2)
    
    # Therefore ∫ ξ^2 |ĝ|^2 dξ = (1/(2π)) ∫ ξ^2 |P(ξ)|^2 exp(-ξ^2) dξ = (1/(2π)) * x2_norm
    # And ∫ |ĝ|^2 dξ = (1/(2π)) * norm2
    
    # So Δp^2 = x2_norm/norm2 = Δx^2
    
    # This means for any even function of this form, Δx = Δp, so the product is Δx^2.
    
    # So we just need to minimize Δx^2 = x2_norm / norm2
    
    dp2 = x2_norm / norm2
    uncertainty_product = dx2 * dp2  # = dx2^2 since dx2 = dp2
    
    return uncertainty_product

def construct():
    """Find optimal coefficients to minimize the uncertainty product."""
    start_time = time.time()
    
    # Initial guess from the seed
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_uncertainty(*best_coeffs)
    if best_val is None:
        best_val = np.inf
    
    # Try some random initializations
    rng = np.random.default_rng(42)
    
    for i in range(50):
        if time.time() > DEADLINE - 30:
            break
        
        # Random initialization around the seed
        coeffs = best_coeffs + rng.normal(0, 0.1, 3)
        val = _compute_uncertainty(*coeffs)
        if val is not None and val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()
    
    # Local optimization
    def objective(x):
        if time.time() > DEADLINE - 10:
            return np.inf
        return _compute_uncertainty(x[0], x[1], x[2])
    
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() > DEADLINE - 20:
            break
        try:
            result = minimize(objective, best_coeffs, method=method, 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
