import numpy as np
import scipy.optimize as opt
import time

# Pre-compute exact integrals for the analytic solution
# Denominator terms: <Hn, Hn> = sqrt(pi) * 2^n * n!
import math

def factorial(n):
    return math.factorial(n)

# Coefficients for H0, H4, H8, H12 (highest power to lowest, or lowest to highest?)
# Let's use lowest to highest for numpy poly1d or manual convolution.
# H0 = 1
# H4 = 12 - 48x^2 + 16x^4
# H8 = 1680 - 13440x^2 + 13440x^4 - 3840x^6 + 256x^8
# H12 = 4790016 - 14515200x^2 + 14515200x^4 - 6635520x^6 + 1290240x^8 - 122880x^10 + 4096x^12

H0_coeffs = [1.0]
H4_coeffs = [12.0, 0.0, -48.0, 0.0, 16.0]
H8_coeffs = [1680.0, 0.0, -13440.0, 0.0, 13440.0, 0.0, -3840.0, 0.0, 256.0]
H12_coeffs = [4790016.0, 0.0, -14515200.0, 0.0, 14515200.0, 0.0, -6635520.0, 0.0, 1290240.0, 0.0, -122880.0, 0.0, 4096.0]

# Values at 0 for constraint P(0)=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0 # Wait, in the seed code H12_0 was 665280.0?
# Let's check H12(0).
# H12(x) = 4096x^12 - ... + 4790016.
# The seed code said: H12_0 = 665280.0.
# Let's re-read the seed code carefully.
# def H12(x): return 4096*x**12 - 122880*x**10 + 1290240*x**8 - 6635520*x**6 + 14515200*x**4 - 14515200*x**2 + 4790016
# H12(0) = 4790016.
# Why did the seed code use 665280.0?
# Maybe the Hermite polynomials are normalized differently?
# Standard Physicists' Hermite polynomials: H_0=1, H_1=2x, H_2=4x^2-2...
# Let's check H4(0) = 12. Seed code: H4_0 = 12.0. Correct.
# H8(0) = 1680. Seed code: H8_0 = 1680.0. Correct.
# H12(0) = 4790016. Seed code: H12_0 = 665280.0.
# 4790016 / 665280 = 7.2. This is not an integer.
# Is it possible the seed code used a different definition?
# Or maybe I copied the H12 polynomial wrong?
# Let's check the recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}.
# H0 = 1
# H1 = 2x
# H2 = 4x^2 - 2
# H3 = 8x^3 - 12x
# H4 = 16x^4 - 48x^2 + 12
# H5 = 32x^5 - 160x^3 + 120x
# H6 = 64x^6 - 480x^4 + 720x^2 - 120
# H7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H8 = 256x^8 - 3840x^6 + 13440x^4 - 13440x^2 + 1680
# H9 = 512x^9 - 10752x^7 + 60480x^5 - 120960x^3 + 60480x
# H10 = 1024x^10 - 30720x^8 + 241920x^6 - 604800x^4 + 403200x^2 - 30240
# H11 = 2048x^11 - 81920x^9 + 806400x^7 - 2721600x^5 + 3024000x^3 - 1209600x
# H12 = 4096x^12 - 184320x^10 + 2304000x^8 - 9676800x^6 + 14515200x^4 - 7257600x^2 + 4790016?
# Let's re-calculate H12(0) from recurrence.
# H12(0) = -2*11*H10(0) = -22 * (-30240) = 665280.
# Ah! My manual expansion of H12 was wrong.
# Let's re-verify the coefficients in the seed code.
# Seed code H12: 4096*x**12 - 122880*x**10 + 1290240*x**8 - 6635520*x**6 + 14515200*x**4 - 14515200*x**2 + 4790016
# My recurrence H12: 4096x^12 - 184320x^10 ...
# The seed code coefficients for H12 seem incorrect for standard Hermite polynomials?
# Or maybe they are not standard?
# Let's check H4 in seed: 16x^4 - 48x^2 + 12. Matches standard.
# H8 in seed: 256x^8 - 3840x^6 + 13440x^4 - 13440x^2 + 1680. Matches standard.
# H12 in seed: 4096x^12 - 122880x^10 + 1290240x^8 - 6635520x^6 + 14515200x^4 - 14515200x^2 + 4790016.
# Let's check H12(0) with seed coeffs: 4790016.
# But seed code says `H12_0 = 665280.0`.
# This is a contradiction in the seed program!
# The `compute_d` function uses `H12_0 = 665280.0`.
# But the `H12` function returns a polynomial that evaluates to 4790016 at 0.
# This means the seed program is inconsistent.
# However, the task says "Keep the function names and signatures...".
# If I use the analytic method, I must be consistent.
# Which one is correct for the "Uncertainty inequality" context?
# Usually, these problems use standard Hermite polynomials.
# If the seed program has a bug, fixing it might improve the score.
# But wait, the score is 0.7763. The inspirations have 1.0.
# Maybe the inspirations fixed this inconsistency?
# If I use the analytic method, I should use the CORRECT Hermite polynomials.
# Let's assume the standard Hermite polynomials are intended.
# I will use the standard coefficients derived from recurrence.

# Recalculated Standard Hermite Coeffs (lowest to highest):
# H0: [1]
# H4: [12, 0, -48, 0, 16]
# H8: [1680, 0, -13440, 0, 13440, 0, -3840, 0, 256]
# H12:
# H10(0) = -30240
# H12(0) = -22 * H10(0) = 665280
# Let's get full H12 coeffs.
# I will generate them programmatically in the solution to be safe.

def get_hermite_coeffs(n):
    # Returns list of coeffs [c0, c1, ..., cn] for Hn(x)
    # H0 = 1
    # H1 = 2x
    # H_{n+1} = 2x H_n - 2n H_{n-1}
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    h_prev = [1.0] # H0
    h_curr = [0.0, 2.0] # H1
    
    for k in range(2, n + 1):
        # H_k = 2x H_{k-1} - 2(k-1) H_{k-2}
        # 2x H_{k-1} shifts h_curr by 1 and multiplies by 2
        term1 = [0.0] + [2.0 * c for c in h_curr]
        # -2(k-1) H_{k-2} multiplies h_prev by -2(k-1)
        term2 = [-2.0 * (k-1) * c for c in h_prev]
        
        # Ensure same length
        max_len = max(len(term1), len(term2))
        while len(term1) < max_len: term1.append(0.0)
        while len(term2) < max_len: term2.append(0.0)
        
        h_next = [t1 + t2 for t1, t2 in zip(term1, term2)]
        
        h_prev = h_curr
        h_curr = h_next
        
    return h_curr

# Pre-compute
H0_c = get_hermite_coeffs(0)
H4_c = get_hermite_coeffs(4)
H8_c = get_hermite_coeffs(8)
H12_c = get_hermite_coeffs(12)

# Orthogonality constants
# <Hn, Hn> = sqrt(pi) * 2^n * n!
import math
sqrt_pi = math.sqrt(math.pi)
norm_H0 = sqrt_pi * (2**0) * factorial(0)
norm_H4 = sqrt_pi * (2**4) * factorial(4)
norm_H8 = sqrt_pi * (2**8) * factorial(8)
norm_H12 = sqrt_pi * (2**12) * factorial(12)

# Integral of x^(2m) e^{-2x^2}
# I_{2m} = (2m-1)!! / (2^m * 2^m) * sqrt(pi/2) ?
# Formula: int x^{2m} e^{-alpha x^2} dx = (2m-1)!! / (2^m alpha^m) * sqrt(pi/alpha)
# Here alpha = 2.
# I_{2m} = (2m-1)!! / (2^m * 2^m) * sqrt(pi/2) = (2m-1)!! / 4^m * sqrt(pi/2)

def double_factorial(n):
    if n < 0: return 1
    if n == 0: return 1
    if n == 1: return 1
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result

sqrt_pi_over_2 = math.sqrt(math.pi / 2.0)
# Pre-compute integrals for powers up to 48
# Max power in P^4 is 4*12 = 48.
# We need I_k for k=0, 2, ..., 48.
integrals = {}
for m in range(0, 25): # 2m up to 48
    k = 2 * m
    if m == 0:
        integrals[k] = sqrt_pi_over_2
    else:
        integrals[k] = double_factorial(2*m - 1) / (4.0 ** m) * sqrt_pi_over_2

def objective_analytic(params):
    a, b, c = params
    d = -(a * H0_c[0] + b * H4_c[0] + c * H8_c[0]) / H12_c[0]
    
    # Construct P(x) coefficients
    # P = a H0 + b H4 + c H8 + d H12
    # Length 13
    p_coeffs = [0.0] * 13
    for i in range(len(H0_c)): p_coeffs[i] += a * H0_c[i]
    for i in range(len(H4_c)): p_coeffs[i] += b * H4_c[i]
    for i in range(len(H8_c)): p_coeffs[i] += c * H8_c[i]
    for i in range(len(H12_c)): p_coeffs[i] += d * H12_c[i]
    
    # Compute P(x)^4 coefficients
    # Use numpy polynomial multiplication
    # np.poly1d uses highest to lowest. Let's stick to lowest to highest and use convolution.
    # Or just use np.polynomial.polynomial.polyfromroots? No, poly1d is easier if we reverse.
    # Let's use manual convolution or numpy convolve.
    # p_coeffs is lowest to highest.
    # q = p * p * p * p
    q = np.convolve(p_coeffs, p_coeffs)
    q = np.convolve(q, p_coeffs)
    q = np.convolve(q, p_coeffs)
    
    # Compute Numerator
    num = 0.0
    for k, coeff in enumerate(q):
        if k % 2 == 0:
            num += coeff * integrals[k]
            
    # Compute Denominator
    den = a*a*norm_H0 + b*b*norm_H4 + c*c*norm_H8 + d*d*norm_H12
    
    if den <= 0:
        return 1e10
        
    return num / den

def construct():
    start_time = time.time()
    deadline = start_time + 190.0
    
    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective_analytic(x0)
    best_params = x0.copy()
    
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Phase 1: Differential Evolution
    try:
        if time.time() < deadline - 60.0:
            de_res = opt.differential_evolution(
                objective_analytic, 
                bounds,
                seed=42,
                maxiter=500, # Can increase since it's fast
                tol=1e-10,
                polish=False,
                update_interval=10
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B
    try:
        if time.time() < deadline - 20.0:
            lbfgs_res = opt.minimize(
                objective_analytic, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if lbfgs_res.fun < best_score:
                best_score = lbfgs_res.fun
                best_params = lbfgs_res.x.copy()
    except Exception:
        pass
        
    return np.array(best_params)
