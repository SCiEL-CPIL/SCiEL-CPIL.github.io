"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
import warnings
warnings.filterwarnings('ignore')

def _eval_g_and_fft(c0, c4, c8, x):
    """
    Evaluate g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0)=0.
    Also returns the Fourier transform G(k) analytically.
    """
    # Coefficients for H0, H4, H8, H12
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 75600x^2 + 10395
    
    # At x=0:
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 10395
    # So c0 + 3*c4 + 105*c8 + 10395*c12 = 0
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
    
    # Build polynomials
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 75600x^2 + 10395
    
    x2 = x*x
    H0 = np.ones_like(x)
    H4 = x2*x2 - 6.0*x2 + 3.0
    H8 = x2**4 - 28.0*x2**3 + 210.0*x2**2 - 420.0*x2 + 105.0
    H12 = x2**6 - 60.0*x2**5 + 1260.0*x2**4 - 12600.0*x2**3 + 55440.0*x2**2 - 75600.0*x2 + 10395.0
    
    g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-0.5*x2)
    
    # Fourier transform: for f(x) = p(x) exp(-x^2/2), the Fourier transform is (2π)^(-1/2) * q(k) exp(-k^2/2)
    # where q(k) is related to p via the Hermite transform.
    # Actually, for the standard normalization where F[f](k) = (1/sqrt(2π)) ∫ f(x) e^{-ikx} dx,
    # and if f(x) = Σ a_n H_n(x) e^{-x^2/2}, then F[f](k) = Σ a_n (-i)^n H_n(k) e^{-k^2/2} / sqrt(2π)
    # Wait, let me be more careful. The standard result is:
    # If f(x) = H_n(x) e^{-x^2/2}, then F[f](k) = (-i)^n H_n(k) e^{-k^2/2} / sqrt(2π)
    # So G(k) = (1/sqrt(2π)) * (c0*H0(k) + c4*(-i)^4 H4(k) + c8*(-i)^8 H8(k) + c12*(-i)^12 H12(k)) * exp(-k^2/2)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So G(k) = (1/sqrt(2π)) * (c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)) * exp(-k^2/2)
    
    k = x  # We evaluate at the same grid points for symmetry
    k2 = k*k
    H0k = np.ones_like(k)
    H4k = k2*k2 - 6.0*k2 + 3.0
    H8k = k2**4 - 28.0*k2**3 + 210.0*k2**2 - 420.0*k2 + 105.0
    H12k = k2**6 - 60.0*k2**5 + 1260.0*k2**4 - 12600.0*k2**3 + 55440.0*k2**2 - 75600.0*k2 + 10395.0
    
    G = (c0*H0k + c4*H4k + c8*H8k + c12*H12k) * np.exp(-0.5*k2) / np.sqrt(2.0*np.pi)
    
    return g, G

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 = A(f) * A(F[f]) where
    A(f) = ||f||_1 / ||f||_2 and A(F[f]) = ||F[f]||_1 / ||F[f]||_2
    """
    # Use adaptive quadrature for L1 and L2 norms
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
    
    def g_func(x):
        x2 = x*x
        H0 = 1.0
        H4 = x2*x2 - 6.0*x2 + 3.0
        H8 = x2**4 - 28.0*x2**3 + 210.0*x2**2 - 420.0*x2 + 105.0
        H12 = x2**6 - 60.0*x2**5 + 1260.0*x2**4 - 12600.0*x2**3 + 55440.0*x2**2 - 75600.0*x2 + 10395.0
        return (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-0.5*x2)
    
    def G_func(k):
        k2 = k*k
        H0k = 1.0
        H4k = k2*k2 - 6.0*k2 + 3.0
        H8k = k2**4 - 28.0*k2**3 + 210.0*k2**2 - 420.0*k2 + 105.0
        H12k = k2**6 - 60.0*k2**5 + 1260.0*k2**4 - 12600.0*k2**3 + 55440.0*k2**2 - 75600.0*k2 + 10395.0
        return (c0*H0k + c4*H4k + c8*H8k + c12*H12k) * np.exp(-0.5*k2) / np.sqrt(2.0*np.pi)
    
    # Compute ||g||_1, ||g||_2, ||G||_1, ||G||_2
    try:
        l1_g, _ = quad(lambda x: abs(g_func(x)), -10, 10, limit=100)
        l2_g, _ = quad(lambda x: g_func(x)**2, -10, 10, limit=100)
        l1_G, _ = quad(lambda k: abs(G_func(k)), -10, 10, limit=100)
        l2_G, _ = quad(lambda k: G_func(k)**2, -10, 10, limit=100)
    except Exception:
        return 1e10
    
    if l2_g <= 0 or l2_G <= 0:
        return 1e10
    
    A_g = l1_g / np.sqrt(l2_g)
    A_G = l1_G / np.sqrt(l2_G)
    
    return A_g * A_G

def _objective(c):
    c0, c4, c8 = c[0], c[1], c[2]
    try:
        return _compute_bound(c0, c4, c8)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308])
    
    best_score = _objective(x0)
    best_coeffs = x0.copy()
    
    # Local optimization with Nelder-Mead
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 60.0:
        return best_coeffs
    
    # Basin hopping for global search
    try:
        res_bh = basinhopping(_objective, x0=best_coeffs, 
                              niter=30, 
                              stepsize=0.5, 
                              minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100}},
                              seed=42)
        if res_bh.fun < best_score:
            best_score = res_bh.fun
            best_coeffs = res_bh.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 20.0:
        return best_coeffs
    
    # Final polish with high precision
    try:
        res_polish = minimize(_objective, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res_polish.fun < best_score:
            best_score = res_polish.fun
            best_coeffs = res_polish.x.copy()
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
