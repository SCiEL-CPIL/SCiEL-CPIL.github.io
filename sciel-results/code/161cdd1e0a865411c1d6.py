"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def objective(params):
        c0, c4, c8 = params
        
        # Enforce g(0) = 0 => c12 is determined
        # c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Construct g(x)
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
        
        # Compute norms
        # L2 norm
        norm2_sq = np.sum(g**2) * dx
        if norm2_sq < 1e-10:
            return 1e10
            
        # L4 norm
        norm4 = np.sum(np.abs(g)**4) * dx
        norm4_val = norm4 ** 0.25
        
        # Fourier Transform
        # Using FFT. The continuous FT is approx by sum g(x) exp(-2pi i x xi) dx
        # We need the xi grid.
        # For FFT of length N, frequencies are k/N * 1/dx ? No.
        # Standard FFT gives DFT. 
        # Let's use scipy.fft for better handling or just np.fft.
        # np.fft.fft(g) * dx approximates the FT at frequencies k / (N*dx) * 2*pi?
        # Actually, let's just compute the L2 and L4 norms of the FT numerically using the FFT.
        
        G = np.fft.fft(g) * dx
        # The frequency spacing is 1/(N*dx)
        # The norm in frequency space: sum |G_k|^2 * dxi
        # dxi = 1 / (N * dx)
        # Note: The FFT output is ordered [0, 1, ..., N/2, -N/2+1, ...]
        # We don't need to shift for norms.
        
        # L2 norm of G
        norm2_hat_sq = np.sum(np.abs(G)**2) / (len(g) * dx) # Wait, Plancherel says ||g||_2 = ||G||_2 if FT is unitary?
        # Let's check scaling.
        # Continuous: ||f||_2^2 = ||hat{f}||_2^2.
        # Discrete approx: sum |g|^2 dx = sum |G|^2 dxi.
        # So norm2_hat_sq should be norm2_sq.
        
        # L4 norm of G
        norm4_hat = np.sum(np.abs(G)**4) * (1 / (len(g) * dx))
        norm4_hat_val = norm4_hat ** 0.25
        
        ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
        
        return ratio

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Use Nelder-Mead for robustness without gradient
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
    
    if res.success:
        return res.x
    else:
        return x0

# EVOLVE-BLOCK-END
