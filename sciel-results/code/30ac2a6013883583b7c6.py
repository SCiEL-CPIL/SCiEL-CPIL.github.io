"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import sqrt, pi

def construct():
    """
    Construct the optimal coefficients for the Hermite polynomial combination.
    Returns:
        np.array([a0, a4, a8])
    """
    start_time = time.time()
    deadline = start_time + 180  # 180s limit, keep safety margin
    
    # Hermite polynomials (probabilists' convention? No, physicists' usually used in this context)
    # Standard Hermite polynomials H_n(x)
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 42x^10 + 840x^8 - 7560x^6 + 30240x^4 - 30240x^2 + 10395
    
    # The test function is g(x) = P(x) * exp(-x^2/2) ? Or exp(-x^2)?
    # Let's check the standard form. Usually uncertainty principle uses exp(-x^2/2) or similar.
    # Goncalves et al. 2017 uses g(x) = (1 + a x^2 + b x^4) exp(-x^2/2) ? No, it's Hermite.
    # Let's assume the standard physicists' Hermite polynomials H_n(x) and Gaussian exp(-x^2/2).
    # Wait, the Fourier transform of exp(-x^2/2) is sqrt(2pi) exp(-k^2/2).
    # Let's derive the moments for g(x) = (a0 H_0 + a4 H_4 + a8 H_8 + a12 H_12) exp(-x^2/2).
    # Condition g(0) = 0 implies a0*1 + a4*3 + a8*105 + a12*10395 = 0.
    # So a12 = -(a0 + 3a4 + 105a8) / 10395.
    
    # We need to minimize the upper bound on A(f) * A(F(f)).
    # A(f) is typically defined as sqrt(<x^2> - <x>^2) * sqrt(<p^2> - <p>^2) ?
    # Or is it the variance product? The prompt says "upper bound on C4".
    # In the context of the uncertainty principle, the product of variances is often denoted.
    # Let's compute the variance of x and the variance of the Fourier variable (k).
    
    # Let's define the function to compute the objective value given a0, a4, a8.
    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        
        # We need to compute <x^2> and <k^2> for the probability density |g(x)|^2 / ||g||^2.
        # g(x) = P(x) exp(-x^2/2).
        # |g(x)|^2 = P(x)^2 exp(-x^2).
        # Norm^2 = integral P(x)^2 exp(-x^2) dx.
        
        # To do this efficiently, we can use the orthogonality of Hermite polynomials?
        # No, P(x) is a sum of H_n. P(x)^2 will have products H_n H_m.
        # H_n(x) H_m(x) can be expanded in terms of H_k(x).
        # This might be complex to do symbolically.
        # Alternative: Numerical integration.
        # But we need to do this many times.
        # Let's use the fact that H_n(x) exp(-x^2/2) are eigenfunctions of the harmonic oscillator.
        # Let phi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) exp(-x^2/2).
        # Then g(x) = sum c_n phi_n(x), where c_n are related to a_n.
        # Specifically, a_n H_n(x) exp(-x^2/2) = a_n sqrt(2^n n! sqrt(pi)) phi_n(x).
        # Let b_n = a_n sqrt(2^n n! sqrt(pi)).
        # Then g(x) = sum b_n phi_n(x).
        # Since phi_n are orthonormal, ||g||^2 = sum b_n^2.
        # <x^2> = sum_{n,m} b_n b_m <phi_n | x^2 | phi_m>.
        # The operator x^2 in the harmonic oscillator basis has non-zero matrix elements only between n and n, n±2.
        # x = sqrt(1/2) (a + a^†).
        # x^2 = 1/2 (a^2 + a†^2 + a a† + a† a) = 1/2 (a^2 + a†^2 + 2a†a + 1).
        # <n | x^2 | m> is non-zero if m = n, n±2.
        # <n | x^2 | n> = n + 1/2.
        # <n | x^2 | n+2> = sqrt((n+1)(n+2)/2).
        # <n | x^2 | n-2> = sqrt(n(n-1)/2).
        
        # Similarly for <k^2>. The Fourier transform of phi_n(x) is (-i)^n phi_n(k) (up to normalization).
        # Actually, the Fourier transform of the harmonic oscillator eigenfunctions are themselves (up to phase).
        # So <k^2> in the Fourier domain is the same as <x^2> in the position domain?
        # Wait, the uncertainty principle is about the variances of x and p (or k).
        # For the harmonic oscillator eigenstates, <x^2> = <p^2> = n + 1/2.
        # But here we have a superposition of states n=0, 4, 8, 12.
        # So <x^2> and <k^2> will be different because the cross terms will differ?
        # No, the Fourier transform of phi_n is (-i)^n phi_n.
        # So |hat{g}(k)|^2 = |g(k)|^2 (if we define the FT properly).
        # Wait, if g(x) = sum b_n phi_n(x), then hat{g}(k) = sum b_n (-i)^n phi_n(k).
        # So the coefficients in the k-domain are b'_n = b_n (-i)^n.
        # The magnitudes |b'_n|^2 are the same as |b_n|^2.
        # The cross terms in <k^2> will involve b'_n b'_m^* = b_n b_m^* (-i)^n (-i)^{-m} = b_n b_m^* (-i)^{n-m}.
        # The cross terms in <x^2> involve b_n b_m^*.
        # So <x^2> and <k^2> will generally be different.
        
        # Let's compute <x^2> and <k^2> using this basis expansion.
        # n values: 0, 4, 8, 12.
        # b_n = a_n sqrt(2^n n! sqrt(pi)).
        
        n_vals = [0, 4, 8, 12]
        a_vals = [a0, a4, a8, a12]
        
        # Compute b_n
        b = np.zeros(4)
        for i, n in enumerate(n_vals):
            b[i] = a_vals[i] * sqrt(2**n * np.math.factorial(n) * sqrt(pi))
            
        norm_sq = np.sum(b**2)
        if norm_sq < 1e-12:
            return 1e10  # Avoid division by zero
            
        # Matrix elements of x^2
        # <n | x^2 | m>
        # We only need elements between n, m in {0, 4, 8, 12}.
        # Non-zero if |n-m| is 0 or 2.
        # Our n values are 0, 4, 8, 12. Differences are 4, 8, 12.
        # None of the differences are 0 or 2!
        # So the off-diagonal terms <n | x^2 | m> are zero for n != m.
        # This means <x^2> = sum b_n^2 <n | x^2 | n> / norm_sq = sum b_n^2 (n + 1/2) / norm_sq.
        
        # Wait, is this correct?
        # x^2 connects n to n, n±2.
        # Our states are 0, 4, 8, 12.
        # 0 connects to 0, 2. (2 is not in our set)
        # 4 connects to 2, 4, 6. (2, 6 not in our set)
        # 8 connects to 6, 8, 10. (6, 10 not in our set)
        # 12 connects to 10, 12, 14. (10, 14 not in our set)
        # So indeed, all off-diagonal terms are zero.
        # This implies <x^2> is just the weighted average of (n+1/2).
        
        # What about <k^2>?
        # The coefficients in k-domain are b'_n = b_n (-i)^n.
        # The matrix elements <n | k^2 | m> are the same as <n | x^2 | m> because k and x have the same algebraic structure in the Fourier domain (they are related by a unitary transformation that maps the basis to itself up to phases).
        # Specifically, the operator k^2 in the k-basis has the same matrix elements as x^2 in the x-basis.
        # So <k^2> = sum |b'_n|^2 (n + 1/2) / norm_sq = sum b_n^2 (n + 1/2) / norm_sq.
        # This would imply <x^2> = <k^2>, which is not generally true for a superposition.
        # Where is the error?
        # The error is in the assumption that the matrix elements are the same.
        # Let's re-evaluate.
        # g(x) = sum b_n phi_n(x).
        # hat{g}(k) = sum b_n (-i)^n phi_n(k).
        # Let c_n = b_n (-i)^n.
        # <k^2> = sum_{n,m} c_n c_m^* <phi_n | k^2 | phi_m>.
        # The operator k^2 in the k-domain is the same as the operator x^2 in the x-domain, BUT the basis states are phi_n(k).
        # The matrix element <phi_n | k^2 | phi_m> is indeed the same as <phi_n | x^2 | phi_m>.
        # So <k^2> = sum c_n c_m^* M_{nm}, where M_{nm} = <phi_n | x^2 | phi_m>.
        # Since M_{nm} is zero unless n=m or |n-m|=2, and our n values differ by 4,8,12, M_{nm} is zero for n!=m.
        # So <k^2> = sum c_n^2 M_{nn} / norm_sq = sum b_n^2 (n+1/2) / norm_sq.
        # This implies <x^2> = <k^2>.
        # This is a known result for states that are superpositions of harmonic oscillator eigenstates with n differing by more than 2?
        # No, that's not right. The variance depends on the cross terms.
        # If the cross terms are zero, then the variance is just the average of the eigenvalues.
        # Let's check with a simple example: psi = (phi_0 + phi_2)/sqrt(2).
        # <x^2> = (1/2)(<0|x^2|0> + <2|x^2|2> + <0|x^2|2> + <2|x^2|0>)
        # = (1/2)(1/2 + 5/2 + sqrt(1) + sqrt(1)) = (1/2)(3 + 2) = 2.5.
        # Weighted average of eigenvalues: (1/2)(0.5) + (1/2)(2.5) = 1.5.
        # So the cross terms matter.
        # In our case, the differences are 4, 8, 12.
        # x^2 only connects n to n±2.
        # So for n=0, it connects to 2. But 2 is not in our set.
        # For n=4, it connects to 2, 6. Not in our set.
        # So the cross terms ARE zero for our specific set of n values {0, 4, 8, 12}.
        # This means <x^2> = sum b_n^2 (n+0.5) / norm_sq.
        # And similarly <k^2> = sum b_n^2 (n+0.5) / norm_sq.
        # So <x^2> = <k^2>.
        # This means the product <x^2><k^2> is just (<x^2>)^2.
        # And the uncertainty product is sqrt(<x^2><k^2>) = <x^2> (assuming <x>=0, <k>=0).
        # So we just need to minimize <x^2> = sum b_n^2 (n+0.5) / sum b_n^2.
        
        # b_n^2 = a_n^2 2^n n! sqrt(pi).
        # Let w_n = a_n^2 2^n n!.
        # Then <x^2> = sum w_n (n+0.5) / sum w_n.
        
        # We want to minimize this ratio.
        # The constraint is a12 = -(a0 + 3a4 + 105a8) / 10395.
        # This is a non-linear optimization problem in a0, a4, a8.
        
        # Let's compute the objective function based on this derivation.
        w = np.zeros(4)
        for i, n in enumerate(n_vals):
            w[i] = a_vals[i]**2 * 2**n * np.math.factorial(n)
            
        num = np.sum(w * (np.array(n_vals) + 0.5))
        den = np.sum(w)
        
        if den < 1e-12:
            return 1e10
            
        return num / den

    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_val = 1e10
    best_x = x0
    
    # Try a few local optimizations from different starting points
    start_points = [
        x0,
        np.array([1, 1, 1]),
        np.array([-1, -1, -1]),
        np.array([1, -1, 1]),
        np.array([-1, 1, -1]),
        np.array([0.1, 0.1, 0.1]),
        np.array([-0.1, -0.1, -0.1]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
            
    # If local optimization didn't find something better, try differential evolution
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-8, seed=42)
            if de_res.success and de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x
        except:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
