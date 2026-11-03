"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE = 200.0  # Safety margin before 210s kill

def _hermite_poly(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return 2 * x
    else:
        # Use recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
        H_nm2 = np.ones_like(x, dtype=float)
        H_nm1 = 2 * x
        for k in range(1, n):
            H_n = 2 * x * H_nm1 - 2 * k * H_nm2
            H_nm2 = H_nm1
            H_nm1 = H_n
        return H_nm1

def _compute_C4(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F̂f) for the Hermite-Gaussian test function.
    
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x²/2)
    where c12 is chosen so that g(0) = 0.
    
    The bound C4 typically involves:
    C4 = (∫|g|⁴ dx)^(1/2) * (∫|ĝ|⁴ dx)^(1/2) or similar
    """
    # Check deadline
    if time.time() - _START_TIME > DEADLINE:
        return np.inf
    
    # Compute c12 from constraint g(0) = 0
    # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 9453600
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 9453600.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Numerically compute the integrals
    # Use a fine grid for numerical integration
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials at x
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    # Gaussian factor
    gauss = np.exp(-x**2 / 2)
    
    # g(x)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # |g|⁴ integral
    g4 = np.abs(g)**4
    integral_g4 = np.trapz(g4, x)
    
    # Fourier transform of g
    # ĝ(ξ) = (1/√(2π)) ∫ g(x) e^{-iξx} dx
    # For numerical computation, use FFT or direct integration
    # Since g is real and even (all Hermite polynomials of even degree are even),
    # ĝ is also real and even
    # Use direct numerical integration for the Fourier transform
    xi = np.linspace(-15, 15, 20001)
    dxi = xi[1] - xi[0]
    
    # Compute Fourier transform numerically
    # ĝ(ξ) = (1/√(2π)) ∫ g(x) cos(ξx) dx (since g is even)
    # This is a 2D integral - might be slow. Let's use a smaller grid.
    
    # Actually, for efficiency, let's use a coarser grid for the Fourier transform
    x_ft = np.linspace(-12, 12, 4001)
    dx_ft = x_ft[1] - x_ft[0]
    
    H0_ft = _hermite_poly(0, x_ft)
    H4_ft = _hermite_poly(4, x_ft)
    H8_ft = _hermite_poly(8, x_ft)
    H12_ft = _hermite_poly(12, x_ft)
    
    gauss_ft = np.exp(-x_ft**2 / 2)
    g_ft = (c0 * H0_ft + c4 * H4_ft + c8 * H8_ft + c12 * H12_ft) * gauss_ft
    
    # Compute Fourier transform using matrix multiplication
    # ĝ(ξ) = (1/√(2π)) ∫ g(x) e^{-iξx} dx
    # For even g: ĝ(ξ) = (1/√(2π)) ∫ g(x) cos(ξx) dx
    xi_ft = np.linspace(-12, 12, 4001)
    
    # Build cosine matrix
    cos_matrix = np.cos(np.outer(xi_ft, x_ft))
    
    # ĝ(ξ) = (1/√(2π)) * cos_matrix @ (g_ft * dx_ft)
    g_hat = (1 / np.sqrt(2 * np.pi)) * (cos_matrix @ (g_ft * dx_ft))
    
    # |ĝ|⁴ integral
    g_hat4 = np.abs(g_hat)**4
    integral_g_hat4 = np.trapz(g_hat4, xi_ft)
    
    # The upper bound C4
    # Based on uncertainty inequality literature, C4 likely involves:
    # C4 = (∫|g|⁴)^(1/2) * (∫|ĝ|⁴)^(1/2)
    # Or it could be a normalized version
    C4 = np.sqrt(integral_g4) * np.sqrt(integral_g_hat4)
    
    return C4

def construct():
    """
    Return three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 weight c12 is fixed so that g(0) = 0.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start from the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try to compute the initial value
    try:
        f0 = _compute_C4(*x0)
    except Exception:
        f0 = np.inf
    
    # If the initial computation failed or took too long, return the prior
    if time.time() - _START_TIME > DEADLINE - 10:
        return x0
    
    # Use multiple optimization strategies
    best_x = x0.copy()
    best_f = f0
    
    if np.isfinite(f0):
        # Try Nelder-Mead
        try:
            res = minimize(_compute_C4, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success and res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass
        
        # Check deadline
        if time.time() - _START_TIME > DEADLINE - 30:
            return best_x
        
        # Try BFGS with numerical gradient
        try:
            res = minimize(_compute_C4, best_x, method='BFGS',
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.success and res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass
        
        # Check deadline
        if time.time() - _START_TIME > DEADLINE - 15:
            return best_x
        
        # Try L-BFGS-B
        try:
            res = minimize(_compute_C4, best_x, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.success and res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass
    
    # Final check
    if time.time() - _START_TIME > DEADLINE - 5:
        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
