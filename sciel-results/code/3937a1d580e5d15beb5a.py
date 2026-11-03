"""Optimized program using active search to minimize the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _hermite_powers(x):
    """
    Returns powers of x up to 12 needed for H0..H12 integrals.
    We need integrals of x^k * e^{-x^2} for k up to 24 (since g^2 has e^{-x^2}).
    Actually, g(x) = P(x) e^{-x^2/2}.
    g(x)^2 = P(x)^2 e^{-x^2}.
    We need moments \int x^n P(x)^2 e^{-x^2} dx.
    """
    pass

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    """
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    # a0 should be 0 by construction
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    # We need I_k = \int x^k P(x)^2 e^{-x^2} dx
    # P(x)^2 = (a12 x^12 + ... + a0)^2
    # This results in a polynomial of degree 24.
    # Let Q(x) = P(x)^2 = \sum_{j=0}^{12} b_{2j} x^{2j}
    # We need integrals \int x^{2j} e^{-x^2} dx = 0.5 * Gamma(j+0.5)
    
    # Calculate coefficients b_{2j} of P(x)^2
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # p = [a0, a2, a4, a6, a8, a10, a12]
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    # q[k] = sum_{i+j=k} p[i]p[j] for x^{2k}
    q = np.convolve(p, p)
    # q has length 13, indices 0 to 12 corresponding to x^0 to x^24
    
    # Integrate x^{2k} e^{-x^2}
    # \int_{-inf}^{inf} x^{2k} e^{-x^2} dx = \sqrt{\pi} (2k-1)!! = \sqrt{\pi} 2^k k!
    # Let's compute these factors
    factors = np.zeros(13)
    for k in range(13):
        # (2k-1)!! = 2^k * k!
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    # I[k] = \int x^{2k} P(x)^2 e^{-x^2} dx
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
    
    # Variance <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    x4 = I[2] / I[0]
    
    # The uncertainty product for this specific Gaussian-weighted polynomial
    # The task mentions "Fourier-invariant". 
    # For g(x) = P(x)e^{-x^2/2}, the Fourier transform is also of the form Q(\xi)e^{-\xi^2/2}.
    # The product A(g)A(\hat{g}) is related to the moments.
    # Standard Heisenberg: sqrt(<x^2> - <x>^2) * sqrt(<xi^2> - <xi>^2).
    # Since g is even, <x>=0.
    # For Fourier transform, <xi^2> = <(-i d/dx)^2> ? No, Parseval.
    # \int |\hat{g}|^2 \xi^2 d\xi = \int |g''|^2 dx / (2\pi)^2 ?
    # Actually, for the specific "uncertainty inequality" context in harmonic analysis (like Beurling-Selberg or similar), 
    # the bound C4 often refers to the constant in the inequality.
    # However, the prompt says "reduce the upper bound on A(f)*A(Fourier(f))".
    # And "score is ... upper bound on C4".
    # If the seed score is 1.0 and better scores are < 1.0 (improvement), and value is lower is better.
    # The inspirations achieved 0.75.
    # Let's assume the objective is to minimize the Heisenberg product for this family, 
    # or a specific bound derived from it.
    # Given the Hermite basis, the Gaussian $e^{-x^2/2}$ has product 1/2.
    # Adding Hermite terms increases the spread.
    # We want to find coefficients that minimize the product.
    
    # Let's use the standard product: sqrt(x2) * sqrt(xi2)
    # For g = P e^{-x^2/2}, \hat{g} is related to P.
    # It turns out that for this class, the product is simply sqrt(<x^2> <x^4> / <x^2>^2 ...)? No.
    # Let's look at the "C4" hint. 
    # In some contexts, C4 is the 4th moment constant.
    # But let's stick to the most likely objective: Minimizing the uncertainty product.
    # For even functions, A(f) = sqrt(<x^2>).
    # A(fhat) = sqrt(<xi^2>).
    # By Parseval, <xi^2> = (1/4\pi^2) \int |g''(x)|^2 dx.
    
    # Let's compute \int (P'' e^{-x^2/2} + 2 P' (-x/2) e^{-x^2/2} + P (x^2/2 - 1/2) e^{-x^2/2})^2 ... this is complex.
    # However, there is a known result for $g(x) = P(x) e^{-x^2/2}$.
    # The Fourier transform is $\hat{g}(\xi) = \frac{1}{\sqrt{2\pi}} \int P(x) e^{-x^2/2} e^{-ix\xi} dx$.
    # This is equivalent to evaluating the generating function.
    # Alternatively, note that $x^2 g(x) = (P x^2) e^{-x^2/2}$.
    # The operator $x$ and $D = d/dx$ satisfy $[x, D] = 1$.
    # The uncertainty product is minimized by Gaussian.
    # We are tuning the polynomial to get "closest" to Gaussian behavior or minimize the product.
    
    # Let's assume the objective function is the Heisenberg product $P(g) = \sqrt{\langle x^2 \rangle \langle \xi^2 \rangle}$.
    # We can compute $\langle \xi^2 \rangle$ using the derivative moments.
    # $\langle \xi^2 \rangle = \frac{1}{\|g\|^2} \int |\hat{g}(\xi)|^2 \xi^2 d\xi = \frac{1}{\|g\|^2} \frac{1}{4\pi^2} \int |g''(x)|^2 dx$.
    
    # Calculate g''(x)
    # g = P e^{-x^2/2}
    # g' = (P' - xP) e^{-x^2/2}
    # g'' = (P'' - xP' - P - x(P' - xP)) e^{-x^2/2}
    #      = (P'' - 2xP' + (x^2 - 1)P) e^{-x^2/2}
    
    # Let R(x) = P''(x) - 2xP'(x) + (x^2 - 1)P(x)
    # We need \int R(x)^2 e^{-x^2} dx.
    
    # P(x) = \sum p_i x^{2i}
    # P'(x) = \sum 2i p_i x^{2i-1}
    # P''(x) = \sum 2i(2i-1) p_i x^{2i-2}
    
    # We need the polynomial R(x) coefficients.
    # P is even. P' is odd. P'' is even.
    # -2xP' is even. (x^2-1)P is even.
    # So R(x) is an even polynomial.
    
    # Let's construct R(x) coefficients r_j for x^{2j}.
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # P'(x) = \sum_{i=1}^6 2i p_i x^{2i-1}
    # -2x P'(x) = \sum_{i=1}^6 -4i p_i x^{2i}
    # P''(x) = \sum_{i=1}^6 2i(2i-1) p_i x^{2i-2} = \sum_{j=0}^5 2(j+1)(2j+1) p_{j+1} x^{2j}
    # (x^2-1)P(x) = \sum_{i=0}^6 p_i x^{2i+2} - \sum_{i=0}^6 p_i x^{2i}
    #              = \sum_{j=1}^7 p_{j-1} x^{2j} - \sum_{j=0}^6 p_j x^{2j}
    
    # Coefficient r_j of x^{2j} in R(x):
    # From P'': term is 2(j+1)(2j+1) p_{j+1} (for j=0..5, 0 for j=6)
    # From -2xP': term is -4j p_j (for j=1..6, 0 for j=0)
    # From x^2 P: term is p_{j-1} (for j=1..7, 0 for j=0)
    # From -P: term is -p_j (for j=0..6, 0 for j=7)
    
    r = np.zeros(8) # indices 0..7
    for j in range(8):
        val = 0.0
        # P'' contribution
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        # -2xP' contribution
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        # x^2 P contribution
        if j >= 1 and j <= 7:
            val += p[j-1]
        # -P contribution
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    # Now compute \int R(x)^2 e^{-x^2} dx
    # R(x) = \sum_{j=0}^7 r_j x^{2j}
    # R(x)^2 = \sum_{k=0}^{14} s_k x^{2k}
    # s = convolve(r, r)
    s = np.convolve(r, r)
    
    # Integrate s_k x^{2k} e^{-x^2}
    # Need factors up to k=14
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    # <xi^2> = norm_R_sq / (4 * pi^2 * norm_sq)
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    # Product
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 # 150s limit, safe margin
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Try L-BFGS-B from seed
    try:
        res = scipy.optimize.minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.success:
            val = res.fun
            if val < best_val:
                best_val = val
                best_coeffs = res.x
    except Exception:
        pass
    
    # If time permits, try a few random restarts to escape local minima
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            # Random restarts in a reasonable range
            # Coeffs are likely small
            r_coeffs = np.random.normal(0, 1, 3)
            try:
                res_r = scipy.optimize.minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    r_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_r.success:
                    val_r = res_r.fun
                    if val_r < best_val:
                        best_val = val_r
                        best_coeffs = res_r.x
            except Exception:
                continue

    return best_coeffs
# EVOLVE-BLOCK-END
