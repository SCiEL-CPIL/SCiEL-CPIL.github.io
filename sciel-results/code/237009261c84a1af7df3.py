import numpy as np
from scipy.optimize import minimize, brentq
from numpy.polynomial.hermite import herm2poly
import time
import math

# Precompute Hermite polynomial coefficients and their derivatives
_h0 = herm2poly([1])
_h4 = herm2poly([0, 0, 0, 0, 1])
_h8 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 1])
_h12 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# Values at x=0 for the P(0)=0 constraint
# herm2poly returns coefficients in ascending order, so index 0 is the constant term
_h0_0 = _h0[0]
_h4_0 = _h4[0]
_h8_0 = _h8[0]
_h12_0 = _h12[0]

# Precompute Gaussian moments I_k = int x^k e^{-x^2} dx for k = 0..30
# I_{2n} = (2n)! / (4^n n!) * sqrt(pi), I_{odd} = 0
_I_moments = np.zeros(31)
for n in range(16):
    k = 2 * n
    _I_moments[k] = np.exp(math.lgamma(2*n + 1) - n * math.log(4) - math.lgamma(n + 1)) * np.sqrt(np.pi)

def _integral_of_poly(coeff):
    """Compute int P(x) e^{-x^2} dx given power-basis coefficients."""
    total = 0.0
    for i in range(len(coeff)):
        if i <= 30:
            total += coeff[i] * _I_moments[i]
    return total

def _get_c12(c0, c4, c8):
    return -(c0 * _h0_0 + c4 * _h4_0 + c8 * _h8_0) / _h12_0

def _compute_bound(c4, c8):
    """Compute the uncertainty bound (B/A)*(C/A) for given c4, c8 with c0=1."""
    c0 = 1.0
    c12 = _get_c12(c0, c4, c8)
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    p_coeff = c0 * _h0 + c4 * _h4 + c8 * _h8 + c12 * _h12
    
    # P^2
    p_sq_coeff = np.convolve(p_coeff, p_coeff)
    
    # Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
    p_prime_coeff = np.polynomial.polynomial.polyder(p_coeff)
    p_double_prime_coeff = np.polynomial.polynomial.polyder(p_prime_coeff)
    
    term1 = -2 * np.concatenate([[0], p_prime_coeff])
    term2 = np.zeros(len(p_coeff) + 2)
    term2[2:] = p_coeff
    term2 -= p_coeff
    
    max_len = max(len(p_double_prime_coeff), len(term1), len(term2))
    p_double_prime_coeff = np.pad(p_double_prime_coeff, (0, max_len - len(p_double_prime_coeff)))
    term1 = np.pad(term1, (0, max_len - len(term1)))
    term2 = np.pad(term2, (0, max_len - len(term2)))
    q_coeff = p_double_prime_coeff + term1 + term2
    
    # Q^2
    q_sq_coeff = np.convolve(q_coeff, q_coeff)
    
    # Integrals
    A = _integral_of_poly(p_sq_coeff)
    B = _integral_of_poly(np.convolve(p_sq_coeff, [0, 0, 1]))
    C = _integral_of_poly(q_sq_coeff)
    
    if A <= 0 or C <= 0:
        return 1e10
    
    bound = (B / A) * (C / A)
    return bound

def _bound_and_grad(params):
    """Compute the uncertainty bound (B/A)*(C/A) and its gradient w.r.t. (c4, c8) with c0=1."""
    c0 = 1.0
    c4, c8 = params
    c12 = _get_c12(c0, c4, c8)
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    p_coeff = c0 * _h0 + c4 * _h4 + c8 * _h8 + c12 * _h12
    
    # P^2
    p_sq_coeff = np.convolve(p_coeff, p_coeff)
    
    # Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
    p_prime_coeff = np.polynomial.polynomial.polyder(p_coeff)
    p_double_prime_coeff = np.polynomial.polynomial.polyder(p_prime_coeff)
    
    term1 = -2 * np.concatenate([[0], p_prime_coeff])
    term2 = np.zeros(len(p_coeff) + 2)
    term2[2:] = p_coeff
    term2 -= p_coeff
    
    max_len = max(len(p_double_prime_coeff), len(term1), len(term2))
    p_double_prime_coeff = np.pad(p_double_prime_coeff, (0, max_len - len(p_double_prime_coeff)))
    term1 = np.pad(term1, (0, max_len - len(term1)))
    term2 = np.pad(term2, (0, max_len - len(term2)))
    q_coeff = p_double_prime_coeff + term1 + term2
    
    # Q^2
    q_sq_coeff = np.convolve(q_coeff, q_coeff)
    
    # Integrals
    A = _integral_of_poly(p_sq_coeff)
    B = _integral_of_poly(np.convolve(p_sq_coeff, [0, 0, 1]))
    C = _integral_of_poly(q_sq_coeff)
    
    if A <= 0 or C <= 0:
        return 1e10, np.array([0.0, 0.0])
    
    # Objective: f = (B/A) * (C/A) = B*C/A^2
    bound = (B / A) * (C / A)
    
    # Compute derivatives using the chain rule
    dc12_dc4 = -_h4_0 / _h12_0
    dc12_dc8 = -_h8_0 / _h12_0
    
    dP_dc4 = _h4 + dc12_dc4 * _h12
    dP_dc8 = _h8 + dc12_dc8 * _h12
    
    def _int_prod_poly(g, h):
        """Compute int G(x) H(x) e^{-x^2} dx given power-basis coefficients."""
        prod = np.convolve(g, h)
        return _integral_of_poly(prod)
    
    P_dP_dc4 = _int_prod_poly(p_coeff, dP_dc4)
    P_dP_dc8 = _int_prod_poly(p_coeff, dP_dc8)
    
    x2_P = np.convolve(p_coeff, [0, 0, 1])
    x2P_dP_dc4 = _int_prod_poly(x2_P, dP_dc4)
    x2P_dP_dc8 = _int_prod_poly(x2_P, dP_dc8)
    
    def _compute_Q_coeff(dP):
        dP_prime = np.polynomial.polynomial.polyder(dP)
        dP_double_prime = np.polynomial.polynomial.polyder(dP_prime)
        t1 = -2 * np.concatenate([[0], dP_prime])
        t2 = np.zeros(len(dP) + 2)
        t2[2:] = dP
        t2 -= dP
        ml = max(len(dP_double_prime), len(t1), len(t2))
        dP_double_prime = np.pad(dP_double_prime, (0, ml - len(dP_double_prime)))
        t1 = np.pad(t1, (0, ml - len(t1)))
        t2 = np.pad(t2, (0, ml - len(t2)))
        return dP_double_prime + t1 + t2
    
    dQ_dc4 = _compute_Q_coeff(dP_dc4)
    dQ_dc8 = _compute_Q_coeff(dP_dc8)
    
    Q_dQ_dc4 = _int_prod_poly(q_coeff, dQ_dc4)
    Q_dQ_dc8 = _int_prod_poly(q_coeff, dQ_dc8)
    
    dA_dc4 = 2 * P_dP_dc4
    dA_dc8 = 2 * P_dP_dc8
    dB_dc4 = 2 * x2P_dP_dc4
    dB_dc8 = 2 * x2P_dP_dc8
    dC_dc4 = 2 * Q_dQ_dc4
    dC_dc8 = 2 * Q_dQ_dc8
    
    A2 = A * A
    A3 = A2 * A
    
    df_dc4 = (dB_dc4 * C + B * dC_dc4) / A2 - 2 * B * C * dA_dc4 / A3
    df_dc8 = (dB_dc8 * C + B * dC_dc8) / A2 - 2 * B * C * dA_dc8 / A3
    
    return bound, np.array([df_dc4, df_dc8])

def construct():
    start_time = time.time()
    deadline = start_time + 170.0

    # Initial guess from seed, scaled so c0=1
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    scale = 1.0 / c0_seed
    c4_init = c4_seed * scale
    c8_init = c8_seed * scale

    best_result = None
    best_score = float('inf')

    # Phase 1: 1D global scan exploiting scale invariance
    # The uncertainty product is scale invariant, so we can set c0=1 and
    # scan over the ratio r = c8/c4. For each r, we solve for c4 that minimizes
    # the 1D function f(c4) = bound(c4, r*c4).
    
    # First, do a coarse grid scan over r to find promising regions
    r_values = np.linspace(-5.0, 5.0, 201)
    r_candidates = []
    r_scores = []
    
    for r in r_values:
        if time.time() > deadline - 30.0:
            break
        # For this r, scan c4 over a reasonable range
        c4_vals = np.linspace(-2.0, 2.0, 50)
        min_score_for_r = float('inf')
        best_c4_for_r = 0.0
        
        for c4 in c4_vals:
            c8 = r * c4
            score = _compute_bound(c4, c8)
            if score < min_score_for_r:
                min_score_for_r = score
                best_c4_for_r = c4
        
        if min_score_for_r < best_score:
            best_score = min_score_for_r
            best_result = np.array([best_c4_for_r, r * best_c4_for_r])
        
        r_candidates.append((r, best_c4_for_r, min_score_for_r))
    
    # Sort by score and keep top candidates
    r_candidates.sort(key=lambda x: x[2])
    top_candidates = r_candidates[:20]
    
    # Phase 2: Refine top candidates with Brent's method in 2D
    starts = []
    
    # Add the seed point
    starts.append([c4_init, c8_init])
    
    # Add top candidates from the 1D scan
    for r, c4, score in top_candidates[:10]:
        starts.append([c4, r * c4])
    
    # Add structured starting points
    structured_starts = [
        [0.1, 0.01],
        [-0.1, -0.01],
        [0.5, 0.05],
        [-0.5, -0.05],
        [0.0, 0.0],
        [0.2, 0.02],
        [-0.2, -0.02],
        [0.3, 0.03],
        [-0.3, -0.03],
        [0.15, 0.015],
        [-0.15, -0.015],
        [0.05, 0.005],
        [-0.05, -0.005],
    ]
    starts.extend(structured_starts)
    
    # Add random starting points
    np.random.seed(42)
    for _ in range(15):
        starts.append([np.random.uniform(-1.5, 1.5), np.random.uniform(-0.2, 0.2)])
    
    # Phase 3: Local optimization from multiple starting points
    for start in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(_bound_and_grad, start, method='L-BFGS-B', 
                          jac=True, options={'gtol': 1e-12, 'maxiter': 300, 'ftol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(_bound_and_grad, best_result, method='L-BFGS-B', 
                          jac=True, options={'gtol': 1e-15, 'maxiter': 500, 'ftol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        return np.array([c0_seed, c4_seed, c8_seed])
    
    c4_opt, c8_opt = best_result
    return np.array([1.0, c4_opt, c8_opt])
