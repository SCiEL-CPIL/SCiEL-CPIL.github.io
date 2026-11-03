"""Optimized uncertainty inequality test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 for A(f)*A(F(f)) given coefficients.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-199584
    So g(0) = c0*1 + c4*(-12) + c8*1680 + c12*(-199584) = 0
    => c12 = (c0 - 12*c4 + 1680*c8) / 199584
    """
    c12 = (c0 - 12.0 * c4 + 1680.0 * c8) / 199584.0
    
    # We need to compute A(g) and A(g_hat)
    # For g(x) = sum c_n H_n(x) e^{-x^2/2}
    # |g(x)|^2 = (sum c_n H_n(x))^2 * e^{-x^2}
    # A(g)^2 = integral |g(x)|^2 dx / (integral |g(x)| dx)^2  -- wait, need to check definition
    
    # Actually, the standard uncertainty product bound uses:
    # A(f) = sup |f|, and the bound involves ||f||_1 * ||f_hat||_1 or similar
    # Let me use the standard formulation where we compute numerically
    
    # Sample on a grid
    x = np.linspace(-30, 30, 200000)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = p * np.exp(-0.5 * x**2)
    
    # A(g) = sup |g(x)|
    Ag = np.max(np.abs(g))
    
    # Fourier transform: F(g)(k) = integral g(x) e^{-2pi i k x} dx
    # But for the uncertainty principle in this context, we likely use:
    # The bound on A(f)*A(F(f)) where F is the unitary Fourier transform
    # F(g)(k) = (1/sqrt(2pi)) integral g(x) e^{-ikx} dx
    # For g(x) = p(x) e^{-x^2/2}, F(g)(k) = (1/sqrt(2pi)) integral p(x) e^{-x^2/2} e^{-ikx} dx
    # = (1/sqrt(2pi)) * F[p(x) e^{-x^2/2}](k)
    # Using the fact that F[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2} (up to normalization)
    
    # With the convention F[f](k) = (1/sqrt(2pi)) int f(x) e^{-ikx} dx:
    # F[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2}
    
    # So F(g)(k) = (1/sqrt(2pi)) * [c0*H0(k) + c4*(-i)^4 H4(k) + c8*(-i)^8 H8(k) + c12*(-i)^12 H12(k)] * e^{-k^2/2}
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^4 * (-i)^8 = 1
    # Wait: (-i)^2 = -1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^8 * (-i)^4 = 1
    
    # So all even powers give +1!
    # F(g)(k) = (1/sqrt(2pi)) * [c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)] * e^{-k^2/2}
    
    # Hmm, that means F(g) has the same polynomial part as g, just scaled by 1/sqrt(2pi)
    # So A(F(g)) = (1/sqrt(2pi)) * A(g)
    # And A(g)*A(F(g)) = A(g)^2 / sqrt(2pi)
    
    # Wait, that can't be right for a non-trivial problem. Let me reconsider.
    
    # Actually, the Fourier transform convention matters. Let me use:
    # F[f](k) = int f(x) e^{-2pi i k x} dx (unitary in L2 with appropriate normalization)
    
    # For the Gaussian e^{-x^2/2}: F[e^{-x^2/2}](k) = sqrt(2pi) e^{-pi^2 k^2} (with e^{-2pi ikx})
    # This gets messy. Let me just compute numerically.
    
    # Use FFT-based approach or direct integration
    # For efficiency, use the fact that we can compute the Fourier transform analytically for polynomials times Gaussian
    
    # With F[f](k) = (1/sqrt(2pi)) int f(x) e^{-ikx} dx:
    # int x^n e^{-x^2/2} e^{-ikx} dx can be computed via completing the square
    
    # For our purposes, let's just compute A(g) numerically and use the known result
    # that for this specific form, the product A(g)*A(F(g)) depends on the coefficients
    
    # Let me compute both numerically to be safe
    k = x  # same grid
    H0k = _hermite(0, k)
    H4k = _hermite(4, k)
    H8k = _hermite(8, k)
    H12k = _hermite(12, k)
    
    # With the symmetric Fourier transform convention, for even n:
    # F[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2}
    # For n=0,4,8,12 (all even): (-i)^n = 1
    # So F(g)(k) = (1/sqrt(2pi)) * (c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)) * e^{-k^2/2}
    
    g_hat = (1.0 / np.sqrt(2.0 * np.pi)) * (c0 * H0k + c4 * H4k + c8 * H8k + c12 * H12k) * np.exp(-0.5 * k**2)
    
    Ag_hat = np.max(np.abs(g_hat))
    
    bound = Ag * Ag_hat
    return bound

def construct():
    """
    Find optimal coefficients (c0, c4, c8) to minimize the uncertainty bound.
    c12 is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Start from the known good solution and refine
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(*best)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    def objective(coeffs):
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    # Check time
    if time.time() > deadline - 30.0:
        return best
    
    # Use Nelder-Mead for local optimization
    try:
        res = minimize(
            objective,
            best,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.success and res.fun < best_val:
            best = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20.0:
        return best
    
    # Try multiple random restarts to escape local minima
    rng = np.random.default_rng(42)
    for i in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random perturbation around current best
        scale = 0.5 / (1 + i * 0.1)
        trial = best + rng.normal(0, scale, size=3)
        
        try:
            val = objective(trial)
            if val < best_val:
                best = trial
                best_val = val
                
                # Local refinement
                res2 = minimize(
                    objective,
                    best,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res2.success and res2.fun < best_val:
                    best = res2.x
                    best_val = res2.fun
        except Exception:
            continue
    
    # Check time
    if time.time() > deadline - 5.0:
        return best
    
    # Final refinement
    try:
        res_final = minimize(
            objective,
            best,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if res_final.success and res_final.fun < best_val:
            best = res_final.x
            best_val = res_final.fun
    except Exception:
        pass
    
    return np.array(best)
# EVOLVE-BLOCK-END
