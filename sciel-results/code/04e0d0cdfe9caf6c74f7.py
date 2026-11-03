"""
Strategy: Re-frame the optimization as a constrained minimization in the Hermite
coefficient space using a different parametrization and a global search (differential
evolution) combined with local refinement. This avoids the local minima traps of
L-BFGS-B from a single seed by exploring the 3D coefficient space more broadly,
while still respecting the g(0)=0 constraint via the derived c12.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

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
    # a0 should be 0 by construction, but we keep it for generality in case of floating point errors
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    # We need I_k = \int x^k P(x)^2 e^{-x^2} dx
    # P(x)^2 is a polynomial of degree 24.
    # Let Q(x) = P(x)^2 = \sum_{j=0}^{12} b_{2j} x^{2j}
    # We need integrals \int x^{2j} e^{-x^2} dx = 0.5 * Gamma(j+0.5) = sqrt(pi) * 2^j * j!
    
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
    
    # Calculate g''(x)
    # g = P e^{-x^2/2}
    # g' = (P' - xP) e^{-x^2/2}
    # g'' = (P'' - 2xP' + (x^2 - 1)P) e^{-x^2/2}
    
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
    deadline = start_time + 180.0 # 180s limit, safe margin
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Strategy: Use Differential Evolution for global search, then refine with L-BFGS-B
    # This is structurally different from the current program which only uses L-BFGS-B
    # from a few random restarts. DE explores the landscape more thoroughly.
    
    # Define bounds for the coefficients. 
    # The seed values are roughly [-1.13, 0.04, 0.0003].
    # Let's search in a wider box to ensure we don't miss a global minimum.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use DE if time permits
    if time.time() < deadline - 30.0:
        try:
            # maxiter is set to allow convergence within time limit
            # popsize is 20 (default is 15)
            res_de = scipy.optimize.differential_evolution(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True # This will run L-BFGS-B on the best result
            )
            if res_de.success:
                val_de = res_de.fun
                if val_de < best_val:
                    best_val = val_de
                    best_coeffs = res_de.x
        except Exception:
            pass
    
    # Final refinement with L-BFGS-B from the best found so far
    if time.time() < deadline - 5.0:
        try:
            res_final = scipy.optimize.minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_final.success:
                val_final = res_final.fun
                if val_final < best_val:
                    best_val = val_final
                    best_coeffs = res_final.x
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
