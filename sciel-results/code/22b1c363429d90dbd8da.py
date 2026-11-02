"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_phys(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h0 = np.ones_like(x)
        h1 = 2 * x
        for k in range(1, n):
            h2 = 2 * x * h1 - 2 * k * h0
            h0, h1 = h1, h2
        return h1

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound for given coefficients."""
    # a12 determined by g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=103950
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 103950.0
    
    # Use quadrature on a fine grid
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    gauss = np.exp(-0.5 * x**2)
    
    H0 = _hermite_phys(0, x)
    H4 = _hermite_phys(4, x)
    H8 = _hermite_phys(8, x)
    H12 = _hermite_phys(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    
    # A(g) = integral of x^2 |g|^2 / integral of |g|^2
    norm2 = np.trapz(np.abs(g)**2, x)
    if norm2 < 1e-30:
        return 1e10
    
    x2_g = np.trapz(x**2 * np.abs(g)**2, x)
    A_g = x2_g / norm2
    
    # Fourier transform: for H_n(x) e^{-x^2/2}, FT is (i)^n H_n(x) e^{-x^2/2}
    # So |FT(g)|^2 = |g|^2 (since |i|^n = 1)
    # But wait, the FT convention matters. With FT(f)(xi) = int f(x) e^{-i x xi} dx
    # FT of H_n(x) e^{-x^2/2} = (sqrt(2*pi)) (i)^n H_n(x) e^{-x^2/2} ... depends on convention
    # For uncertainty, we need A(FT(g)) = int xi^2 |FT(g)(xi)|^2 dxi / int |FT(g)(xi)|^2 dxi
    # By Parseval and the self-similarity, this equals A_g in magnitude for pure Hermite-Gaussians
    # But for linear combinations, the cross terms matter differently in x-space vs xi-space
    
    # Actually for the standard FT convention, FT[H_n(x) e^{-x^2/2}](xi) = (2*pi)^{1/2} (i)^n H_n(xi) e^{-xi^2/2}
    # So |FT(g)(xi)|^2 has the same form as |g(x)|^2 but with the (i)^n phases
    # The magnitude squared is the same! So A(FT(g)) = A(g)
    # That can't be right for the optimization to be non-trivial...
    
    # Let me reconsider. Perhaps the Gaussian is e^{-x^2/4} so that FT is cleaner.
    # Or perhaps the bound involves higher moments.
    
    # Given the seed values and the problem structure, let me try a different approach:
    # The bound C4 might be related to the 4th moment or a specific inequality.
    
    # Let me try: C4 = A(g) * A(FT(g)) where we compute both properly
    # With FT convention: hat(f)(xi) = (1/sqrt(2*pi)) int f(x) e^{-i x xi} dx
    # Then FT[H_n(x) e^{-x^2/2}](xi) = (i)^n H_n(xi) e^{-xi^2/2}
    # |hat(g)(xi)|^2 = |sum a_k (i)^k H_k(xi) e^{-xi^2/2}|^2
    # This is NOT the same as |g(x)|^2 because of the (i)^k phases!
    
    # So we need to compute both separately
    g_mag2 = np.abs(g)**2
    norm2 = np.trapz(g_mag2, x)
    if norm2 < 1e-30:
        return 1e10
    
    A_g = np.trapz(x**2 * g_mag2, x) / norm2
    
    # Fourier side: hat_g(xi) = sum a_k (i)^k H_k(xi) e^{-xi^2/2}
    # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
    phases = np.array([1.0, 1.0, 1.0, -1.0])  # for k=0,4,8,12
    coeffs = np.array([a0, a4, a8, a12])
    
    hat_g = (coeffs[0] * phases[0] * H0 + coeffs[1] * phases[1] * H4 + 
             coeffs[2] * phases[2] * H8 + coeffs[3] * phases[3] * H12) * gauss
    
    hat_g_mag2 = np.abs(hat_g)**2
    hat_norm2 = np.trapz(hat_g_mag2, x)
    if hat_norm2 < 1e-30:
        return 1e10
    
    A_hat_g = np.trapz(x**2 * hat_g_mag2, x) / hat_norm2
    
    return A_g * A_hat_g

def construct():
    """Return three coefficients of the Hermite combination."""
    start_time = time.time()
    deadline = start_time + 190  # 190s deadline, 20s safety margin
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(*seed)
    best_params = seed.copy()
    
    # Multi-start optimization
    n_starts = 50
    np.random.seed(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 30:  # Leave 30s for final polish
            break
        
        # Random start near the seed
        if i == 0:
            x0 = seed.copy()
        else:
            x0 = seed + np.random.normal(0, 0.5, size=3)
        
        try:
            res = minimize(
                lambda p: _compute_bound(p[0], p[1], p[2]),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda p: _compute_bound(p[0], p[1], p[2]),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
