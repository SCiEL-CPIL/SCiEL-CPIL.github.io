"""Optimized program: Analytic integrals for Hermite-Gaussian uncertainty product."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 5.0

# Indices of the Hermite polynomials used
_IDXS = [0, 4, 8, 12]
_N_IDX = len(_IDXKS)

def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_nm2 = np.ones_like(x)
        h_nm1 = 2.0 * x
        for k in range(2, n + 1):
            h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
            h_nm2, h_nm1 = h_nm1, h_n
        return h_nm1

# Precompute Hermite polynomials up to degree 13 (needed for x*H_12)
_MAX_DEG = 13
# We work with coefficients in the Hermite basis for the polynomial part
# We need to compute:
# 1. L_k = integral H_k(x) exp(-x^2/2) dx
# 2. G_ij = integral (2i H_{i-1} - x H_i)(2j H_{j-1} - x H_j) exp(-x^2) dx
# 3. D_k = 2*pi * sqrt(pi) * 2^k * k!

# We will use the property:
# x H_n(x) = 0.5 * H_{n+1}(x) + n * H_{n-1}(x)
# So we can express any polynomial P(x) as sum c_m H_m(x).
# Then integral P(x) Q(x) exp(-x^2) dx = sum_m c_m^P c_m^Q sqrt(pi) 2^m m!

def _poly_to_hermite_coeffs(poly_coeffs):
    """
    Convert standard power basis coefficients [c0, c1, c2, ...] to Hermite basis coefficients [h0, h1, h2, ...]
    H_n(x) are physicist's Hermite polynomials.
    We need to express P(x) = sum c_k x^k as sum h_m H_m(x).
    """
    # This is a bit complex. It's easier to use the recurrence relation for the integral directly
    # or use a numerical integration on a small grid for the precomputation step, 
    # but analytic is better for precision.
    # Actually, since we only have 4 basis functions, we can just compute the integrals numerically 
    # on a very fine grid once. The speed of the optimization loop is what matters.
    # The precomputation cost is negligible compared to 180s.
    pass

# Let's just use high-precision numerical integration for the precomputation of L and G.
# This avoids complex symbolic algebra.
_X_PRE = np.linspace(-20.0, 20.0, 10001)
_DW = _X_PRE[1] - _X_PRE[0]
_W1 = np.exp(-0.5 * _X_PRE * _X_PRE)
_W2 = np.exp(-_X_PRE * _X_PRE)

# Precompute H_n for n=0..13
_H = {}
for n in range(14):
    _H[n] = _hermite(n, _X_PRE)

# 1. Compute L_k = integral H_k(x) exp(-x^2/2) dx
_L = np.zeros(4)
for i, k in enumerate(_IDXKS):
    _L[i] = np.trapz(_H[k] * _W1, _X_PRE)

# 2. Compute G_ij = integral B_i'(x) B_j'(x) dx
# B_k'(x) = (2k H_{k-1}(x) - x H_k(x)) exp(-x^2/2)
# Let P_k(x) = 2k H_{k-1}(x) - x H_k(x)
# G_ij = integral P_i(x) P_j(x) exp(-x^2) dx
_P = np.zeros((4, len(_X_PRE)))
for i, k in enumerate(_IDXKS):
    if k == 0:
        # H_{-1} is 0. P_0 = -x H_0 = -x
        _P[i] = -_X_PRE
    else:
        _P[i] = 2.0 * k * _H[k-1] - _X_PRE * _H[k]

_G = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _G[i, j] = np.trapz(_P[i] * _P[j] * _W2, _X_PRE)

# 3. Compute D_k = 2*pi * sqrt(pi) * 2^k * k!
_D = np.zeros(4)
for i, k in enumerate(_IDXKS):
    _D[i] = 2.0 * np.pi * np.sqrt(np.pi) * (2.0 ** k) * np.math.factorial(k)

# Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# H0(0) = 1
# H4(0) = 12 * (-1)^2 / 2! ? No, H4(x) = 16x^4 - 48x^2 + 12. H4(0) = 12.
# H8(0) = 168 * (-1)^4 / 4! ? H8(x) = 256x^8 - 4480x^6 + 30240x^4 - 80640x^2 + 10080. H8(0) = 10080.
# H12(0) = 10080 * 12 * 10 * 8 * 6 * 4 * 2 / 6! ? 
# General formula: H_{2n}(0) = (-1)^n (2n)! / n!
# H4(0) = (-1)^2 4! / 2! = 24/2 = 12. Correct.
# H8(0) = (-1)^4 8! / 4! = 40320 / 24 = 1680. Wait.
# Let's check H8(0) from polynomial: 10080.
# 8! = 40320. 4! = 24. 40320/24 = 1680.
# Why the discrepancy?
# Physicist's Hermite: H_4(x) = 16x^4 - 48x^2 + 12.
# H_8(x) = 256x^8 - 4480x^6 + 30240x^4 - 80640x^2 + 10080.
# Formula H_{2n}(0) = (-1)^n (2n)! / n!
# n=4: 8! / 4! = 1680.
# My polynomial H8(0) is 10080.
# 10080 / 1680 = 6.
# Ah, the recurrence H_{n+1} = 2x H_n - 2n H_{n-1}.
# H0=1, H1=2x, H2=4x^2-2, H3=8x^3-12x, H4=16x^4-48x^2+12.
# H4(0)=12. Formula: 4!/2! = 12. Correct.
# H8(0) from recurrence:
# H5 = 2x(16x^4-48x^2+12) - 8(8x^3-12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x
# H6 = 2x(32x^5 - 160x^3 + 120x) - 10(16x^4-48x^2+12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120
# H7 = 2x(64x^6 - 480x^4 + 720x^2 - 120) - 12(32x^5 - 160x^3 + 120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H8 = 2x(128x^7 - 1344x^5 + 3360x^3 - 1680x) - 14(64x^6 - 480x^4 + 720x^2 - 120)
# = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
# = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# H8(0) = 1680.
# So my previous polynomial was wrong. The recurrence is the truth.
# H12(0):
# Formula: 12! / 6! = 479001600 / 720 = 665280.
# Sign: (-1)^6 = 1. So H12(0) = 665280.

_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# a12 = -(a0*1 + a4*12 + a8*1680) / 665280

def _objective(ratios):
    r4, r8 = ratios
    a0 = 1.0
    a4 = r4
    a8 = r8
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    a = np.array([a0, a4, a8, a12])
    
    # I1 = | sum a_k L_k |
    I1 = abs(np.dot(a, _L))
    
    # I2 = sqrt( a^T G a )
    I2_sq = np.dot(a, _G @ a)
    if I2_sq < 0: I2_sq = 0.0
    I2 = np.sqrt(I2_sq)
    
    # I3 = sqrt( sum a_k^2 D_k )
    I3_sq = np.dot(a * a, _D)
    if I3_sq < 0: I3_sq = 0.0
    I3 = np.sqrt(I3_sq)
    
    return I1 * I2 * I3

def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    # Convert to ratios for initial guess
    r4_init = best_coeffs[1] / best_coeffs[0]
    r8_init = best_coeffs[2] / best_coeffs[0]
    
    best_val = _objective([r4_init, r8_init])
    best_r4 = r4_init
    best_r8 = r8_init
    
    # Grid search to find a good basin
    # r4 is around -0.035, r8 around -0.0002
    r4_range = np.linspace(-0.5, 0.5, 20)
    r8_range = np.linspace(-0.02, 0.02, 20)
    
    for r4 in r4_range:
        if time.time() > _DEADLINE - _MARGIN: break
        for r8 in r8_range:
            val = _objective([r4, r8])
            if val < best_val:
                best_val = val
                best_r4 = r4
                best_r8 = r8
    
    # Local optimization
    res = minimize(_objective, [best_r4, best_r8], method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_r4 = res.x[0]
        best_r8 = res.x[1]
        
    # Polish
    res = minimize(_objective, [best_r4, best_r8], method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_r4 = res.x[0]
        best_r8 = res.x[1]
        
    # Return coefficients
    result_coeffs = np.array([1.0, best_r4, best_r8])
    
    # Safety check
    if np.all(np.isfinite(result_coeffs)):
        return result_coeffs
    else:
        return best_coeffs
# EVOLVE-BLOCK-END
