"""
Rewrite: numerical optimization of the upper bound on the 4th-moment uncertainty
product A(f)*A(F(f)) for a Hermite-Gaussian test function g = c0*H0 + c4*H4 + c8*H8 + c12*H12
with g(0) = 0 (c12 determined from c0, c4, c8).

The objective is the product of the 4th-moment position uncertainty and the 4th-moment
frequency uncertainty:
    B = ||x^2 g||_2 * ||xi^2 g_hat||_2
subject to ||g||_2 = 1 and g(0) = 0.

We minimize B over (c0, c4, c8) using scipy.optimize.
"""

import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: 180s hard limit, use 170s for safety
T_START = time.time()
DEADLINE = 170.0

def time_left():
    return DEADLINE - (time.time() - T_START)

def hermite_h(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h0 = np.ones_like(x)
        h1 = 2.0 * x
        for k in range(2, n + 1):
            h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def gaussian(x):
    """Standard Gaussian exp(-x^2/2)."""
    return np.exp(-0.5 * x * x)

def g_function(x, c0, c4, c8):
    """Construct g(x) = c0*H0*phi + c4*H4*phi + c8*H8*phi + c12*H12*phi, with g(0)=0."""
    phi = gaussian(x)
    h0 = hermite_h(0, x)
    h4 = hermite_h(4, x)
    h8 = hermite_h(8, x)
    h12 = hermite_h(12, x)
    
    # g(0) = 0 condition:
    # H0(0) = 1, H4(0) = -12, H8(0) = 1680, H12(0) = -47900 (physicist's convention)
    # Let me verify: H_n(0) for even n = (-1)^(n/2) * n! / (n/2)!
    # H0(0) = 1
    # H4(0) = (-1)^2 * 4! / 2! = 24/2 = 12? No wait.
    # Actually physicist's H_n(0): H_{2k}(0) = (-1)^k (2k)! / k!
    # H4(0) = (-1)^2 * 4! / 2! = 1 * 24/2 = 12
    # H8(0) = (-1)^4 * 8! / 4! = 1 * 40320/24 = 1680
    # H12(0) = (-1)^6 * 12! / 6! = 1 * 479001600/720 = 665280? Let me recompute.
    # 12! = 479001600, 6! = 720, so 12!/6! = 479001600/720 = 665280
    # But the sign: (-1)^6 = 1, so H12(0) = 665280
    # Hmm, let me double check with known values.
    # Actually, I recall H_4(0) = -12 in some conventions. Let me just compute numerically.
    
    h0_0 = hermite_h(0, np.array([0.0]))[0]
    h4_0 = hermite_h(4, np.array([0.0]))[0]
    h8_0 = hermite_h(8, np.array([0.0]))[0]
    h12_0 = hermite_h(12, np.array([0.0]))[0]
    
    # c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
    # c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    return (c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12) * phi

def norm2_sq(f, x, dx):
    """Compute ||f||_2^2 via numerical integration."""
    return np.trapz(f * f, x, dx=dx)

def objective(params):
    c0, c4, c8 = params
    if time_left() < 5.0:
        return 1e10
    
    # Use a fine grid for accurate integration
    x = np.linspace(-15, 15, 10000)
    dx = x[1] - x[0]
    
    g = g_function(x, c0, c4, c8)
    
    # Normalize
    norm_g_sq = norm2_sq(g, x, dx)
    if norm_g_sq < 1e-15:
        return 1e10
    g_norm = g / np.sqrt(norm_g_sq)
    
    # Position 4th moment: ||x^2 g||_2
    x2_g = x**2 * g_norm
    pos_4th_sq = norm2_sq(x2_g, x, dx)
    
    # Frequency 4th moment: ||xi^2 g_hat||_2
    # g_hat(xi) = (1/sqrt(2*pi)) * integral g(x) exp(-i*xi*x) dx
    # By Plancherel, ||xi^2 g_hat||_2^2 = (1/4*pi^2) * ||d^2/dx^2 g||_2^2 (up to constants)
    # Actually, more carefully: if F(omega) = (1/sqrt(2*pi)) int f(x) e^{-i omega x} dx
    # then ||omega^2 F||_2 = (1/sqrt(2*pi)) ||f''||_2 (by integration by parts, boundary terms vanish for Gaussian)
    # Wait: omega * F(omega) corresponds to (1/sqrt(2*pi)) int i x f(x) e^{-i omega x} dx = (1/sqrt(2*pi)) F{i f}(omega)... 
    # Let me use: Fourier transform convention F(omega) = (1/sqrt(2*pi)) int f(x) e^{-i omega x} dx
    # Then omega^n F(omega) = (1/sqrt(2*pi)) int (i x)^n f(x) e^{-i omega x} dx = (i/sqrt(2*pi))^n ... no
    # Actually: d/domega F(omega) = (1/sqrt(2*pi)) int (-i x) f(x) e^{-i omega x} dx
    # So omega F(omega) = i * (1/sqrt(2*pi)) int x f(x) e^{-i omega x} dx = i * F[x f](omega)
    # Thus ||omega F||_2 = ||F[x f]||_2 = ||x f||_2 (by Plancherel)
    # And omega^2 F(omega) = i * F[x (i x f)] = -F[x^2 f]... let me be more careful.
    # omega^2 F = omega * (omega F) = omega * i F[x f] = i * F[x (x f)] = i * F[x^2 f]
    # So ||omega^2 F||_2 = ||F[x^2 f]||_2 = ||x^2 f||_2
    # That gives the same as position! That can't be right for a non-trivial problem.
    
    # Let me reconsider. The "uncertainty" in the 4th moment context is likely:
    # A(f) = ||x^2 f||_2 / ||f||_2^(something) and A(F(f)) = ||xi^2 F(f)||_2 / ||F(f)||_2^(something)
    # But if ||xi^2 F||_2 = ||x^2 f||_2 by the above, then the product is just ||x^2 f||_2^2,
    # which is minimized by spreading out, giving no interesting constraint.
    
    # I think I'm confusing myself. Let me use the standard definition:
    # The 4th moment of position: <x^4> = ||x^2 f||_2^2 / ||f||_2^2
    # The 4th moment of frequency: <xi^4> = ||xi^2 F||_2^2 / ||F||_2^2
    # By Plancherel, ||F||_2 = ||f||_2
    # And ||xi^2 F||_2^2 = ||d^2 f/dx^2||_2^2 (this is the key identity)
    # Because: xi^2 F(omega) = (1/sqrt(2*pi)) int xi^2 f(x) e^{-i omega x} dx
    # And int xi^2 f(x) e^{-i omega x} dx = - int f''(x) e^{-i omega x} dx (integration by parts twice)
    # So xi^2 F(omega) = -(1/sqrt(2*pi)) int f''(x) e^{-i omega x} dx = -F[f''](omega)
    # Thus ||xi^2 F||_2 = ||F[f'']||_2 = ||f''||_2
    
    # So the objective is ||x^2 g||_2 * ||g''||_2 (with g normalized)
    
    # Compute g'' numerically
    # Use finite differences or spectral method
    # For a smooth Gaussian-like function on a uniform grid, use 2nd order central differences
    dg2 = np.zeros_like(g_norm)
    dg2[1:-1] = (g_norm[2:] - 2*g_norm[1:-1] + g_norm[:-2]) / dx**2
    dg2[0] = dg2[1]
    dg2[-1] = dg2[-2]
    
    freq_4th_sq = norm2_sq(dg2, x, dx)
    
    pos_4th = np.sqrt(pos_4th_sq)
    freq_4th = np.sqrt(freq_4th_sq)
    
    return pos_4th * freq_4th

def run_optimization():
    if time_left() < 10.0:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Seed from the task
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Try multiple starting points
    best_result = None
    best_val = np.inf
    
    starts = [x0,
              [-1.0, 0.05, 0.0003],
              [-1.5, 0.03, 0.0005],
              [-0.8, 0.06, 0.0002],
              [-1.2, 0.035, 0.00035],
              [-1.0, 0.0, 0.0],
              [-2.0, 0.1, 0.001],
              [-0.5, 0.1, 0.001]]
    
    for start in starts:
        if time_left() < 10.0:
            break
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    else:
        return x0

def construct():
    return run_optimization()

if __name__ == "__main__":
    result = construct()
    print("Coefficients (c0, c4, c8):", result)
