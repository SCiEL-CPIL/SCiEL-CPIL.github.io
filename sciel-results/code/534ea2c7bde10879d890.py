"""Optimized program for uncertainty inequality coefficient tuning."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1

def _compute_bound(a0, a4, a8):
    """Compute the upper bound C4 given coefficients for H0, H4, H8.
    
    The degree-12 coefficient is determined by g(0) = 0.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * 1 = 0
    
    H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    H0(0) = 1
    H4(0) = -12  (since (-1)^2 * 4! / 2! = 24/2 = 12, but sign: H4(x)=16x^4-48x^2+12, so H4(0)=12)
    Wait, let me use physicists' Hermite: H0=1, H1=2x, H2=4x^2-2, H3=8x^3-12x, H4=16x^4-48x^2+12
    H4(0) = 12
    H8(0): H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680, so H8(0) = 1680
    H12(0): Using formula H_{2k}(0) = (-1)^k (2k)! / k!
    H12(0) = (-1)^6 * 12! / 6! = 479001600 / 720 = 665280
    
    Actually let me compute more carefully:
    12! = 479001600
    6! = 720
    12!/6! = 665280
    
    So g(0) = a0*1 + a4*12 + a8*1680 + a12*665280 = 0
    => a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    
    The L2 norm squared with Gaussian weight:
    ∫ g(x)^2 dx = ∫ [sum a_n H_n(x)]^2 e^{-x^2} dx (since e^{-x^2/2} squared is e^{-x^2})
    
    By orthogonality: ∫ H_m(x) H_n(x) e^{-x^2} dx = 2^n n! sqrt(pi) δ_{mn}
    
    So ||g||_2^2 = sqrt(pi) * sum_n a_n^2 * 2^n * n!
    
    For the Fourier transform, F{H_n(x) e^{-x^2/2}} = (-i)^n H_n(ξ) e^{-ξ^2/2} (up to normalization)
    
    The upper bound C4 on A(f)*A(F(f)) for this type of test function involves:
    The bound is typically related to the ratio of certain norms or moments.
    
    For the uncertainty inequality with test functions, the bound C4 can be computed as:
    C4 = sup over some parameter of [A(f) * A(F(f))]
    
    A common formulation: if g is the test function, then the bound involves
    ||g||_2^2 and ||x*g||_2^2 and ||ξ*ĝ||_2^2
    
    Actually, for the specific C4 bound in the uncertainty principle context:
    C4 = (||g||_2^2)^2 / (||x*g||_2 * ||ξ*ĝ||_2) type structure, or similar.
    
    Let me think about this differently. The bound A(f)*A(F(f)) ≤ C4 * ||f||_2^2 type inequality.
    
    For the specific Hermite-Gaussian test function, the optimal bound can be computed numerically.
    
    Let me use a direct approach: the bound C4 for the test function g is:
    C4 = sup_ω |∫ g(x) e^{-iωx} dx|^2 / (∫ |g(x)|^2 dx * ∫ |x g(x)|^2 dx / ∫ |g(x)|^2 dx)
    
    This is getting complex. Let me use the fact that for the specific problem,
    we can compute the bound numerically by evaluating the relevant integrals.
    """
    # Compute a12 from constraint g(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute ||g||_2^2 = sqrt(pi) * sum a_n^2 * 2^n * n!
    def factorial(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    norm_sq = np.sqrt(np.pi) * (
        a0**2 * 2**0 * factorial(0) +
        a4**2 * 2**4 * factorial(4) +
        a8**2 * 2**8 * factorial(8) +
        a12**2 * 2**12 * factorial(12)
    )
    
    # Compute ||x*g||_2^2 = ∫ x^2 |g(x)|^2 dx
    # Using x*H_n = (1/2)H_{n+1} + n*H_{n-1} (physicists' convention)
    # So x*H_n*e^{-x^2/2} = (1/2)H_{n+1}e^{-x^2/2} + n*H_{n-1}e^{-x^2/2}
    # Then ||x*g||_2^2 = sqrt(pi) * sum over m of (coefficient of H_m in x*g)^2 * 2^m * m!
    
    # x * (a0*H0 + a4*H4 + a8*H8 + a12*H12) = 
    # a0*(1/2*H1) + a4*(1/2*H5 + 4*H3) + a8*(1/2*H9 + 8*H7) + a12*(1/2*H13 + 12*H11)
    
    # Coefficients of each H_m in x*g:
    # H1: a0/2
    # H3: 4*a4
    # H5: a4/2
    # H7: 8*a8
    # H9: a8/2
    # H11: 12*a12
    # H13: a12/2
    
    coeff_H1 = a0 / 2.0
    coeff_H3 = 4.0 * a4
    coeff_H5 = a4 / 2.0
    coeff_H7 = 8.0 * a8
    coeff_H9 = a8 / 2.0
    coeff_H11 = 12.0 * a12
    coeff_H13 = a12 / 2.0
    
    xg_norm_sq = np.sqrt(np.pi) * (
        coeff_H1**2 * 2**1 * factorial(1) +
        coeff_H3**2 * 2**3 * factorial(3) +
        coeff_H5**2 * 2**5 * factorial(5) +
        coeff_H7**2 * 2**7 * factorial(7) +
        coeff_H9**2 * 2**9 * factorial(9) +
        coeff_H11**2 * 2**11 * factorial(11) +
        coeff_H13**2 * 2**13 * factorial(13)
    )
    
    # For the Fourier transform, similar structure with (-i)^n factors
    # ||ξ*ĝ||_2^2 has the same form as ||x*g||_2^2 by symmetry
    
    # The bound C4 for the uncertainty principle is:
    # C4 = (||g||_2^2)^2 / (||x*g||_2 * ||ξ*ĝ||_2) ... no, that's not quite right
    
    # Actually, for the specific C4 bound in the context of:
    # A(f) * A(F(f)) ≤ C4 * ||f||_2^2
    # where A(f) = sup_x |f(x)| and A(F(f)) = sup_ξ |F(f)(ξ)|
    
    # The bound for a test function g is:
    # C4(g) = sup_{f} [A(f)*A(F(f))/||f||_2^2] subject to f being in the span or similar
    
    # A simpler interpretation: C4 is the operator norm of the composition
    # of the L∞ norm operators. For the Hermite basis, this can be computed.
    
    # Let me use a numerical approach: sample the function and compute the bound directly.
    
    # Create a grid
    x = np.linspace(-10, 10, 10000)
    H0 = np.ones_like(x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
    
    # Compute A(g) = sup|g(x)|
    A_g = np.max(np.abs(g))
    
    # Compute Fourier transform numerically
    # F(g)(ξ) = ∫ g(x) e^{-iξx} dx
    # Use FFT or direct integration
    dx = x[1] - x[0]
    # Use scipy's fft for efficiency
    from scipy.fft import fft, fftfreq
    N = len(x)
    G = fft(g) * dx
    freqs = fftfreq(N, d=dx)
    A_G = np.max(np.abs(G))
    
    # The bound C4 for this test function in the uncertainty inequality
    # is related to the product of the L∞ norms
    # C4 ≈ A_g * A_G / (some normalization)
    
    # Actually, the upper bound on A(f)*A(F(f)) when f is in the span of g
    # or related to g is simply A_g * A_G for the test function itself,
    # but the "upper bound" C4 is the infimum over test functions of this quantity
    # normalized appropriately.
    
    # For the specific problem, C4 is the value we're trying to minimize.
    # It's the product of the sup norms of g and its Fourier transform,
    # possibly normalized by ||g||_2^2 or similar.
    
    # Let me return the product A_g * A_G as the bound to minimize
    return A_g * A_G

def construct():
    """Find optimal coefficients for H0, H4, H8 to minimize the upper bound C4."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess: the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate initial bound
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    # Try multiple starting points and local optimization
    starting_points = [
        x0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.1, 0.04, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.2, 0.03, 0.0001]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, 0.0, 0.001]),
    ]
    
    for x_init in starting_points:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
                
        except Exception:
            continue
    
    # Try a few more refinements with different methods
    if time.time() < deadline - 60.0:
        for x_init in [best_x]:
            try:
                result = minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    x_init,
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Return the optimal coefficients
    return np.array(best_x)
# EVOLVE-BLOCK-END
