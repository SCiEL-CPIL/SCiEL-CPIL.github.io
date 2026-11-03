"""Rewritten program using global optimization for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize


def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    The test function is g(x) = sum_{k in {0,4,8,12}} a_k * H_k(x) * exp(-x^2/2)
    where H_k are physicists' Hermite polynomials (or probabilists', depending on convention).
    
    We minimize the upper bound on A(f)*A(F(fourier(f))) which relates to C4.
    """
    # Use probabilists' Hermite polynomials He_k (standard in Fourier analysis contexts)
    # He_0(x) = 1
    # He_4(x) = x^4 - 6x^2 + 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 45360x^2 + 10395
    
    # The Gaussian weight is exp(-x^2/2), so the L2 inner product with weight exp(-x^2) 
    # (since |g|^2 has exp(-x^2)) corresponds to the standard Gaussian measure.
    
    # For the uncertainty principle with Gaussian weight, the relevant quantity is:
    # A(f) ~ sqrt(<x^2 f, f> / <f,f>) and A(F(f)) ~ sqrt(<k^2 F, F> / <F,F>)
    # By Plancherel, <k^2 F, F> = <(d/dx)^2 f, f> (up to sign/convention)
    
    # The upper bound C4 is related to the product of variances.
    # For a function g = sum a_k He_k, we can compute:
    # <x^2 g, g> and <g'', g> in the Gaussian-weighted space.
    
    # In the probabilists' Hermite basis with weight exp(-x^2/2):
    # The operator x^2 acts as: x^2 He_k = He_{k+2} + k He_k
    # Wait, let me use the correct recurrence. For probabilists' He:
    # x He_n = He_{n+1} + n He_{n-1}
    # So x^2 He_n = x(He_{n+1} + n He_{n-1}) = He_{n+2} + (n+1)He_n + n(He_n + (n-1)He_{n-2})
    # = He_{n+2} + (2n+1)He_n + n(n-1)He_{n-2}
    
    # And d/dx He_n = n He_{n-1}
    # d^2/dx^2 He_n = n(n-1) He_{n-2}
    
    # The inner product <He_i, He_j> = i! delta_{ij} (with weight exp(-x^2/2) dx)
    
    # Let g = a0*He_0 + a4*He_4 + a8*He_8 + a12*He_12
    # With constraint g(0) = 0: a0*1 + a4*3 + a8*105 + a12*10395 = 0
    # So a12 = -(a0 + 3*a4 + 105*a8) / 10395
    
    # Compute <g, g> = a0^2*0! + a4^2*4! + a8^2*8! + a12^2*12!
    # Compute <x^2 g, g>:
    # x^2 He_0 = He_2 + 1*He_0 -> contributes to He_0 and He_2
    # x^2 He_4 = He_6 + 9*He_4 + 12*He_2
    # x^2 He_8 = He_10 + 17*He_8 + 56*He_6 + ... let me recalculate
    # x^2 He_n = He_{n+2} + (2n+1)He_n + n(n-1)He_{n-2}
    # n=0: x^2 He_0 = He_2 + 1*He_0 + 0 = He_2 + He_0
    # n=4: x^2 He_4 = He_6 + 9*He_4 + 12*He_2
    # n=8: x^2 He_8 = He_10 + 17*He_8 + 56*He_6
    # n=12: x^2 He_12 = He_14 + 25*He_12 + 132*He_10
    
    # <x^2 g, g> = sum over i,j of a_i a_j <x^2 He_i, He_j>
    # Only diagonal terms in Hermite basis survive due to orthogonality:
    # <x^2 He_n, He_n> = (2n+1) n!
    # <x^2 He_0, He_0> = 1 * 0! = 1
    # <x^2 He_4, He_4> = 9 * 4! = 9*24 = 216
    # <x^2 He_8, He_8> = 17 * 8! = 17*40320 = 685440
    # <x^2 He_12, He_12> = 25 * 12! = 25*479001600 = 11975040000
    
    # Also off-diagonal: <x^2 He_0, He_2> = 1*2! but He_2 not in our set
    # <x^2 He_4, He_2> = 12*2! but He_2 not in our set
    # <x^2 He_4, He_6> = 1*6! but He_6 not in our set
    # etc. So only diagonal terms matter.
    
    # <g'', g> = sum a_i a_j <He_i'', He_j> = sum a_i^2 i(i-1)(i-2)! for i>=2
    # = a4^2 * 4*3*2! + a8^2 * 8*7*6! + a12^2 * 12*11*10!
    # = a4^2 * 12*2 + a8^2 * 56*720 + a12^2 * 132*3628800
    # = 24*a4^2 + 40320*a8^2 + 479001600*a12^2
    
    # The uncertainty product is bounded by sqrt(<x^2 g, g>/norm^2 * <g'', g>/norm^2)
    # We want to minimize this product.
    
    # Let's define the objective as the product of the two variances.
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        
        # Norm squared
        norm_sq = a0**2 * 1 + a4**2 * 24 + a8**2 * 40320 + a12**2 * 479001600
        
        if norm_sq < 1e-15:
            return 1e15
        
        # x^2 variance numerator: <x^2 g, g>
        x2_num = a0**2 * 1 + a4**2 * 216 + a8**2 * 685440 + a12**2 * 11975040000
        
        # Derivative norm squared numerator: <g', g'>
        g_prime_num = 96*a4**2 + 322560*a8**2 + 5748019200*a12**2
        
        # Product of variances
        return (x2_num / norm_sq) * (g_prime_num / norm_sq)
    
    # Define bounds for the coefficients
    # The scale of the coefficients doesn't matter for the ratio, but for numerical stability
    # and to avoid huge values in a12, we can bound them.
    # a12 is determined by a0, a4, a8.
    # If a0, a4, a8 are O(1), a12 is O(1/10000).
    # Let's search in a reasonable range.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use differential evolution for global optimization
    result = differential_evolution(objective, bounds, maxiter=1000, popsize=15, seed=42)
    
    if result.success:
        return result.x
    else:
        # Fallback to a local refinement if DE fails (unlikely)
        initial_guess = np.array([-1.13, 0.04, 0.0003])
        res_local = minimize(objective, initial_guess, method='Nelder-Mead', 
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        return res_local.x


# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
