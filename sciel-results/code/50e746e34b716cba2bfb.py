import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite

# Precompute Hermite polynomials for degrees 0, 4, 8, 12
# H_n(x) = (-1)^n e^{x^2/2} (d^n/dx^n) e^{-x^2/2}
# We use the standard probabilists' or physicists'? 
# The task mentions "Hermite-polynomial times Gaussian". 
# Standard in this context (Goncalves et al.) usually uses Physicists' Hermite polynomials H_n(x) 
# such that H_n(x) e^{-x^2/2} are the eigenfunctions of the harmonic oscillator (up to scaling).
# Let's verify the values at 0 for Physicists' Hermite polynomials:
# H_0(x) = 1 -> H_0(0) = 1
# H_1(x) = 2x -> H_1(0) = 0
# H_2(x) = 4x^2 - 2 -> H_2(0) = -2
# H_3(x) = 8x^3 - 12x -> H_3(0) = 0
# H_4(x) = 16x^4 - 48x^2 + 12 -> H_4(0) = 12? 
# Wait, standard definition: H_4(x) = 16x^4 - 48x^2 + 12. 
# Let's check the sign. 
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_4(0) = 12.
# However, the seed code implies a specific structure. 
# In many physics contexts, the normalized basis functions are $\psi_n(x) = (2^n n! \sqrt{\pi})^{-1/2} H_n(x) e^{-x^2/2}$.
# The task says "Hermite-polynomial times Gaussian". It doesn't specify normalization.
# The coefficients in the seed are likely tuned for a specific normalization or just the raw polynomial.
# Let's assume the standard Physicists' Hermite polynomials H_n(x).

# Let's double check H_8(0) and H_12(0).
# H_8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 13440.
# H_8(0) = 13440.
# H_12(x) = ... + (-1)^{12/2} * ... 
# General formula for H_{2n}(0) = (-1)^n (2n)!.
# H_4(0) = (-1)^2 * 4! = 24? 
# My previous manual derivation gave 12. Let's re-evaluate.
# Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}.
# H_0 = 1
# H_1 = 2x
# H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2. H_2(0) = -2.
# H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x. H_3(0) = 0.
# H_4 = 2x(8x^3-12x) - 8(4x^2-2) = 16x^4 - 24x^2 - 32x^2 + 16 = 16x^4 - 56x^2 + 16.
# Wait, recurrence is $H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)$.
# H_2(0) = -2.
# H_4(0) = 2(0)H_3(0) - 2(3)H_2(0) = 0 - 6(-2) = 12.
# Okay, so H_4(0) = 12.
# H_6(0) = 2(0)H_5(0) - 2(5)H_4(0) = -10(12) = -120.
# H_8(0) = 2(0)H_7(0) - 2(7)H_6(0) = -14(-120) = 1680.
# H_10(0) = -18(1680) = -30240.
# H_12(0) = -22(-30240) = 665280.

# Let's check the formula $H_{2n}(0) = (-1)^n (2n)!$.
# n=1 (H_2): (-1)^1 * 2! = -2. Correct.
# n=2 (H_4): (-1)^2 * 4! = 24. 
# My recurrence gave 12. Why?
# Ah, the recurrence for Physicists' Hermite is $H_{n+1} = 2x H_n - 2n H_{n-1}$.
# Let's re-calculate H_4 carefully.
# H_2 = 4x^2 - 2.
# H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x.
# H_4 = 2x(8x^3-12x) - 8(4x^2-2) = 16x^4 - 24x^2 - 32x^2 + 16 = 16x^4 - 56x^2 + 16.
# H_4(0) = 16.
# Wait, the formula $H_{2n}(0) = (-1)^n (2n)!$ is for *Probabilists'* Hermite polynomials $He_n(x)$?
# No, for Physicists' $H_n$, the leading coefficient is $2^n$.
# $H_4(x) = 16x^4 - 48x^2 + 12$ is a common form seen in some tables?
# Let's check Wikipedia "Hermite polynomials".
# $H_0 = 1$
# $H_1 = 2x$
# $H_2 = 4x^2 - 2$
# $H_3 = 8x^3 - 12x$
# $H_4 = 16x^4 - 48x^2 + 12$
# Let's check the recurrence for H_4 with this polynomial.
# $2x H_3 - 6 H_2 = 2x(8x^3 - 12x) - 6(4x^2 - 2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12$.
# Okay, my previous manual expansion of $H_4$ had an arithmetic error in the coefficient of $x^2$.
# So $H_4(0) = 12$.
# $H_6(0) = 2(0)H_5 - 10 H_4(0) = -120$.
# $H_8(0) = -14 H_6(0) = 1680$.
# $H_{10}(0) = -18 H_8(0) = -30240$.
# $H_{12}(0) = -22 H_{10}(0) = 665280$.

# The seed coefficients are $c_0 = -1.13, c_4 = 0.04, c_8 = 1/3240 \approx 0.0003086$.
# The constraint is $g(0) = 0$.
# $g(0) = c_0 H_0(0) + c_4 H_4(0) + c_8 H_8(0) + c_{12} H_{12}(0) = 0$.
# $1.13(-1) + 0.04(12) + c_8(1680) + c_{12}(665280) = 0$? 
# Note: $H_0(0)=1$. So $c_0(1) + c_4(12) + c_8(1680) + c_{12}(665280) = 0$.
# $c_{12} = - \frac{c_0 + 12 c_4 + 1680 c_8}{665280}$.

# We will implement the objective function using this constraint.
# We will use a grid for numerical integration to be fast.

def construct():
    start_time = time.time()
    deadline = start_time + 170  # 10s safety margin
    
    # Seed coefficients
    c0_seed = -113.0 / 100.0
    c4_seed = 1.0 / 25.0
    c8_seed = 1.0 / 3240.0
    
    # Grid setup for fast integration
    # Range [-L, L] with L large enough to cover the Gaussian tail.
    # e^{-x^2/2} is negligible for |x| > 10.
    L = 15.0
    N = 5000
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    # Precompute basis functions on the grid
    # H_n(x) e^{-x^2/2}
    # We can compute H_n(x) using recurrence or scipy.special.hermite (which returns poly1d, slow for arrays)
    # Better: use explicit polynomials or recurrence on the grid.
    
    def hermite_gauss(n, x):
        # Compute H_n(x) * exp(-x^2/2) using recurrence
        # H_0(x) = 1
        # H_1(x) = 2x
        # H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x)
        if n == 0:
            return np.exp(-0.5 * x**2)
        if n == 1:
            return 2 * x * np.exp(-0.5 * x**2)
        
        H0 = np.exp(-0.5 * x**2)
        H1 = 2 * x * H0
        Hk_minus_1 = H0
        Hk = H1
        for k in range(1, n):
            Hk_plus_1 = 2 * x * Hk - 2 * k * Hk_minus_1
            Hk_minus_1 = Hk
            Hk = Hk_plus_1
        return Hk

    # Precompute the 4 basis functions
    basis_0 = hermite_gauss(0, x)
    basis_4 = hermite_gauss(4, x)
    basis_8 = hermite_gauss(8, x)
    basis_12 = hermite_gauss(12, x)
    
    # Values at 0 for constraint
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    def objective(params):
        c0, c4, c8 = params
        # Calculate c12 to satisfy g(0) = 0
        # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Compute g(x)
        g = c0 * basis_0 + c4 * basis_4 + c8 * basis_8 + c12 * basis_12
        
        # Compute A(g) = integral |g(x)| dx
        # Using trapezoidal rule
        A_g = np.sum(np.abs(g)) * dx
        
        # Compute Fourier transform g_hat(x)
        # F[g](xi) = sum c_k (-i)^k H_k(xi) e^{-xi^2/2}
        # Note: The basis functions for the Fourier transform are the same shapes, 
        # just with complex coefficients.
        # Since we need |g_hat|, and the basis functions are real, 
        # g_hat(x) = c0*basis_0 - c4*basis_4 - c8*basis_8 + c12*basis_12 ?
        # Let's check phases:
        # F[H_0 e^-x^2/2] = H_0 e^-xi^2/2 (phase 1)
        # F[H_4 e^-x^2/2] = (-i)^4 H_4 e^-xi^2/2 = 1 * H_4 e^-xi^2/2 (phase 1)
        # F[H_8 e^-x^2/2] = (-i)^8 H_8 e^-xi^2/2 = 1 * H_8 e^-xi^2/2 (phase 1)
        # F[H_12 e^-x^2/2] = (-i)^12 H_12 e^-xi^2/2 = 1 * H_12 e^-xi^2/2 (phase 1)
        # Wait, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
        # So all phases are 1 for these specific even degrees divisible by 4.
        # Therefore, g_hat(x) has the same form as g(x) with the same coefficients.
        # A_hat_g = A_g.
        # Is this correct?
        # Let's verify. The Fourier transform of $H_{2m}(x)e^{-x^2/2}$ is $H_{2m}(x)e^{-x^2/2}$ if $2m$ is a multiple of 4?
        # $(-i)^{2m} = (-1)^m$.
        # For m=0 (deg 0): (-1)^0 = 1.
        # For m=2 (deg 4): (-1)^2 = 1.
        # For m=4 (deg 8): (-1)^4 = 1.
        # For m=6 (deg 12): (-1)^6 = 1.
        # Yes, all phases are +1.
        # So A(g) = A(hat{g}).
        # The product is A(g)^2.
        
        return A_g * A_g

    # Initial guess
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    # Run optimization
    best_result = objective(x0)
    best_x = x0
    
    try:
        # Use BFGS for local optimization
        res = minimize(objective, x0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
        
        if res.success and res.fun < best_result:
            best_result = res.fun
            best_x = res.x
            
        # Check time
        if time.time() < deadline:
            # Try a few random perturbations to escape local minima if stuck
            # Or just try L-BFGS-B with bounds
            # The coefficients shouldn't be too large.
            bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            res2 = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
            if res2.success and res2.fun < best_result:
                best_result = res2.fun
                best_x = res2.x
                
    except Exception:
        pass

    # Return the coefficients of H0, H4, H8
    # The function construct() returns the array of 3 coefficients
    return np.array(best_x)

if __name__ == "__main__":
    # This block is for local testing, not required by the evaluator
    pass
