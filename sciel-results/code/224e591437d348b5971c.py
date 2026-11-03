import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time


def construct():
    t_start = time.time()
    deadline = 200.0  # 10s safety margin from 210s
    
    # Hermite polynomial values at 0 for the constraint
    # H_n(0) = 0 for odd n, and for even n=2k: H_{2k}(0) = (-1)^k (2k)! / k!
    H0_0 = 1.0
    H4_0 = 12.0   # H_4(0) = 12
    H8_0 = 1680.0 # H_8(0) = 1680
    H12_0 = 3991680.0  # H_12(0) = 3991680
    
    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def objective(params):
        a0, a4, a8 = params
        a12 = compute_a12(a0, a4, a8)
        
        # Build the polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        # Using explicit coefficients
        # H0(x) = 1
        # H4(x) = 16x^4 - 48x^2 + 12
        # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
        # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 13271040x^4 - 12355200x^2 + 3991680
        
        # Coefficients for powers x^0, x^2, x^4, x^6, x^8, x^10, x^12
        c0 = a0 * 1 + a4 * 12 + a8 * 1680 + a12 * 3991680
        c2 = a4 * (-48) + a8 * (-13440) + a12 * (-12355200)
        c4 = a4 * 16 + a8 * 11520 + a12 * 13271040
        c6 = a8 * (-3072) + a12 * (-5806080)
        c8 = a8 * 256 + a12 * 1032192
        c10 = a12 * (-98304)
        c12 = a12 * 4096
        
        # Note: c0 should be ~0 by construction (P(0)=0)
        
        # Numerically compute the bound
        # g(x) = P(x) * exp(-pi*x^2)
        # We need ||g||_inf and ||g_hat||_inf
        
        # Use a fine grid
        x = np.linspace(-8, 8, 20001)
        x2 = x**2
        x4 = x2**2
        x6 = x4 * x2
        x8 = x4**2
        x10 = x8 * x2
        x12 = x8 * x4
        
        P = c0 + c2*x2 + c4*x4 + c6*x6 + c8*x8 + c10*x10 + c12*x12
        g = P * np.exp(-np.pi * x2)
        
        norm_g = np.max(np.abs(g))
        
        # Compute Fourier transform numerically
        # g_hat(xi) = integral g(x) exp(-2*pi*i*x*xi) dx
        # Use FFT with appropriate scaling
        N = len(x)
        dx = x[1] - x[0]
        
        # FFT gives sum g[n] * exp(-2*pi*i*n*k/N)
        # We need integral, so multiply by dx
        # Also need to handle the frequency axis properly
        
        G = np.fft.fftshift(np.fft.fft(g)) * dx
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        
        norm_g_hat = np.max(np.abs(G))
        
        # The uncertainty product upper bound
        # For the L-infinity version, a common bound involves these norms
        # The task mentions "upper bound on C4" - let's use the product as the objective
        # This is a heuristic; the actual bound may differ
        
        # Actually, for uncertainty principles, the relevant quantity is often
        # related to the ratio or product. Let's use norm_g * norm_g_hat as proxy
        # but we need to be careful about what "upper bound on C4" means
        
        # A simpler interpretation: minimize the product of sup norms
        return norm_g * norm_g_hat
    
    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = None
    best_score = np.inf
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x.copy()
                    
        except Exception as e:
            continue
    
    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113/100, 1/25, 1/3240])
    
    return best_result
