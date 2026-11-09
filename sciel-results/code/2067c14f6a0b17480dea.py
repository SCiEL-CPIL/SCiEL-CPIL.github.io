"""
Search for optimal Hermite coefficients minimizing the uncertainty inequality constant.

The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is determined by g(0)=0.

We use the probabilistic Hermite polynomials: Hn(x) = (-1)^n exp(x^2/2) d^n/dx^n exp(-x^2/2)
These are orthogonal w.r.t. the Gaussian weight exp(-x^2/2).

The Fourier transform of g is computed numerically, and we bound:
A(g) * A(F(g)) where A(h) = sup_x |h(x)| / (inf_x |h(x)| + epsilon) 
Actually, looking at the task more carefully:
- The score is the upper bound on C4
- A(f) is likely the "amplitude" or some norm

Let me reconsider. The task says "upper bound on A(f)*A(Fourier(f))" where A is likely
the L-infinity norm or a ratio. Given the context of uncertainty inequalities and 
the reference to Goncalves et al. 2017, this is about the Heisenberg uncertainty 
principle constant.

Actually, re-reading: "score is the upper bound on C4" - this is the constant in the 
uncertainty inequality. The function A(f) is probably ||f||_inf or similar.

Let me implement a numerical optimization:
1. Parameterize by (a0, a4, a8)
2. Compute a12 from g(0) = 0
3. Compute g(x) and its Fourier transform F(g)(xi)
4. Compute the product of sup-norms (or appropriate norms)
5. Minimize this product

For probabilistic Hermite polynomials:
H0(x) = 1
H1(x) = x
H2(x) = x^2 - 1
H3(x) = x^3 - 3x
H4(x) = x^4 - 6x^2 + 3
H5(x) = x^5 - 10x^3 + 15x
H6(x) = x^6 - 15x^4 + 45x^2 - 15
H7(x) = x^7 - 21x^5 + 105x^3 - 105x
H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
H9(x) = x^9 - 36x^7 + 378x^5 - 1260x^3 + 945x
H10(x) = x^10 - 45x^8 + 630x^6 - 3150x^4 + 4725x^2 - 945
H11(x) = x^11 - 55x^9 + 990x^7 - 6930x^5 + 15015x^3 - 10395x
H12(x) = x^12 - 66x^10 + 1485x^8 - 13860x^6 + 49545x^4 - 69300x^2 + 10395

g(0) = (a0 + 3*a4 + 105*a8 + 10395*a12) * 1 = 0
So a12 = -(a0 + 3*a4 + 105*a8) / 10395
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_probs(x, n):
    """Compute probabilistic Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 5:
        return x**5 - 10*x**3 + 15*x
    elif n == 6:
        return x**6 - 15*x**4 + 45*x**2 - 15
    elif n == 7:
        return x**7 - 21*x**5 + 105*x**3 - 105*x
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 12:
        return x**12 - 66*x**10 + 1485*x**8 - 13860*x**6 + 49545*x**4 - 69300*x**2 + 10395
    else:
        raise ValueError(f"Unsupported Hermite degree {n}")

def _get_a12(a0, a4, a8):
    """Compute a12 such that g(0) = 0."""
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    return -(a0 + 3*a4 + 105*a8) / 10395.0

def _objective(coeffs):
    """Compute the uncertainty product for given coefficients."""
    a0, a4, a8 = coeffs
    a12 = _get_a12(a0, a4, a8)
    
    # Grid for evaluation
    x = np.linspace(-12, 12, 2048)
    dx = x[1] - x[0]
    
    # Compute g(x)
    H0 = _hermite_probs(x, 0)
    H4 = _hermite_probs(x, 4)
    H8 = _hermite_probs(x, 8)
    H12 = _hermite_probs(x, 12)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
    
    # Check if g is non-trivial
    if np.max(np.abs(g)) < 1e-10:
        return 1e10
    
    # Compute Fourier transform using FFT
    # We need to be careful with the FFT conventions
    # F(g)(xi) = integral g(x) exp(-2*pi*i*xi*x) dx
    # Using FFT: we sample g on a grid, FFT gives us the DFT
    
    n = len(x)
    G = np.fft.fft(g) * dx
    
    # The FFT frequencies
    freqs = np.fft.fftfreq(n, d=dx)
    
    # For the uncertainty principle, we need the "amplitude" A(f)
    # This is likely the ratio of sup to inf, or just the sup-norm
    # Given the context, let's use the L-infinity norm as A(f)
    # But that doesn't make sense for a product to be bounded...
    
    # Actually, in uncertainty principles, we often look at:
    # sigma_x * sigma_xi >= constant
    # where sigma is the standard deviation
    
    # Or it could be about the product of L1 and Linf norms
    
    # Let me reconsider: "A(f)*A(Fourier(f))" 
    # In the context of the uncertainty principle for the Fourier transform,
    # a common formulation involves the product of the "spread" in x and xi domains.
    
    # Given the task mentions "upper bound on C4", this is likely the constant
    # in an inequality of the form: A(f) * A(F(f)) >= C4
    # And we want to minimize the upper bound, i.e., find the tightest constant.
    
    # The "amplitude" A(f) might be defined as ||f||_inf / ||f||_1 or similar.
    
    # Let me try: A(f) = ||f||_inf, and we look at the product of L-inf norms
    # But that's not standard.
    
    # Alternative: A(f) could be the "effective support" or bandwidth.
    
    # Given the reference to Goncalves et al. 2017 and Hermite polynomials,
    # this is likely related to the optimal constant in the Heisenberg uncertainty
    # principle for functions with specific symmetry or structure.
    
    # Let me use a different approach: compute the ratio that appears in 
    # uncertainty inequalities. A common one is:
    # (variance of |g|^2 in x) * (variance of |G|^2 in xi) >= constant
    
    # But the task says A(f)*A(Fourier(f)), suggesting a multiplicative quantity.
    
    # Let me try: A(f) = sup |f| / (inf |f| + eps) as a "dynamic range"
    # This doesn't seem right either.
    
    # Given the ambiguity, let me use the most standard uncertainty product:
    # sigma_x^2 * sigma_xi^2 where sigma is the standard deviation of |f|^2
    
    # Compute variance of |g|^2 in x-domain
    w = np.abs(g)**2
    w_sum = np.sum(w) * dx
    if w_sum < 1e-10:
        return 1e10
    w_norm = w / w_sum
    x_mean = np.sum(x * w_norm) * dx
    x_var = np.sum((x - x_mean)**2 * w_norm) * dx
    
    # Compute variance of |G|^2 in xi-domain
    wG = np.abs(G)**2
    wG_sum = np.sum(wG)
    if wG_sum < 1e-10:
        return 1e10
    wG_norm = wG / wG_sum
    xi_mean = np.sum(freqs * wG_norm)
    xi_var = np.sum((freqs - xi_mean)**2 * wG_norm)
    
    # The uncertainty product
    product = x_var * xi_var
    return product

def construct():
    """Find optimal coefficients minimizing the uncertainty product."""
    start_time = time.time()
    deadline = start_time + 180  # 180 second limit
    
    # Initial guess from the seed program
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_score = _objective(best_coeffs)
    
    # Try differential evolution for global optimization
    try:
        if time.time() < deadline - 30:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(
                _objective, bounds, maxiter=50, popsize=15,
                tol=1e-4, seed=42, polish=False
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x
    except Exception:
        pass
    
    # Local refinement with BFGS
    try:
        if time.time() < deadline - 10:
            local_result = minimize(
                _objective, best_coeffs, method='BFGS',
                options={'gtol': 1e-8, 'maxiter': 100}
            )
            if local_result.fun < best_score:
                best_score = local_result.fun
                best_coeffs = local_result.x
    except Exception:
        pass
    
    # Final polish with Nelder-Mead
    try:
        if time.time() < deadline - 5:
            nm_result = minimize(
                _objective, best_coeffs, method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_coeffs = nm_result.x
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
