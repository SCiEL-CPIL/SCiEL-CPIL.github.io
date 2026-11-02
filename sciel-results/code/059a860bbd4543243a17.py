import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
import math

# Pre-compute Hermite polynomial coefficients in ascending order (c0, c1, ..., cn)
# H_0(x) = 1
# H_1(x) = 2x
# H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)

def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    h_nm2 = _hermite_coeffs(n - 2)
    h_nm1 = _hermite_coeffs(n - 1)
    
    # 2x * H_{n-1}(x)
    # H_{n-1} has length n. 2x*H_{n-1} has length n+1.
    term1 = np.zeros(n + 1)
    term1[1:] = 2.0 * h_nm1
    
    # -2(n-1) * H_{n-2}(x)
    term2 = -2.0 * (n - 1) * h_nm2
    if len(term2) < n + 1:
        term2 = np.pad(term2, (0, (n + 1) - len(term2)))
        
    return term1 + term2

# Coefficients for H0, H4, H8, H12
H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

# Pad to length 13 (degree 12) for consistent vector operations
def _pad_coeff(c, length):
    if len(c) >= length:
        return c[:length]
    return np.pad(c, (0, length - len(c)))

H0_p = _pad_coeff(H0_c, 13)
H4_p = _pad_coeff(H4_c, 13)
H8_p = _pad_coeff(H8_c, 13)
H12_p = _pad_coeff(H12_c, 13)

# Values at x=0 (constant terms)
H0_0 = H0_c[0]
H4_0 = H4_c[0]
H8_0 = H8_c[0]
H12_0 = H12_c[0]

# Pre-compute the matrix for P(x) coefficients given (c0, c4, c8, c12)
# P_c = c0*H0_p + c4*H4_p + c8*H8_p + c12*H12_p
# We want P_c as a function of c0, c4, c8.
# c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
# Let v = [c0, c4, c8]
# c12 = -dot(v, H_0_vec) / H12_0, where H_0_vec = [H0_0, H4_0, H8_0]
H_0_vec = np.array([H0_0, H4_0, H8_0])

# Matrix M such that P_c = M @ [c0, c4, c8]
# P_c = c0*(H0_p - (H0_0/H12_0)*H12_p) + c4*(H4_p - (H4_0/H12_0)*H12_p) + c8*(H8_p - (H8_0/H12_0)*H12_p)
M = np.column_stack([
    H0_p - (H0_0 / H12_0) * H12_p,
    H4_p - (H4_0 / H12_0) * H12_p,
    H8_p - (H8_0 / H12_0) * H12_p
])

# Pre-compute the matrix for Q(x) coefficients
# h(x) = x * P(x) * exp(-x^2/2)
# h'(x) = (P(x) + x P'(x) - x^2 P(x)) * exp(-x^2/2)
# Let Q(x) = P(x) + x P'(x) - x^2 P(x)
# We need the coefficients of Q(x) in terms of P_c.
# P_c has length 13 (indices 0..12).
# P'(x) coeffs: P'_c[i] = (i+1)P_c[i+1] for i=0..11. P'_c[12]=0.
# x P'(x) coeffs: shift P'_c right by 1. (x P')_c[i] = P'_c[i-1] for i>=1, 0 for i=0.
# -x^2 P(x) coeffs: shift P_c right by 2 and negate. (-x^2 P)_c[i] = -P_c[i-2] for i>=2, 0 for i<2.

# Let's build a transformation matrix T such that Q_c = T @ P_c
# Q_c[i] = P_c[i] + (i*P_c[i] if i<12 else 0) - (P_c[i-2] if i>=2 else 0)
# Wait, let's be precise with indices.
# P(x) = sum_{k=0}^{12} P_c[k] x^k
# P'(x) = sum_{k=1}^{12} k P_c[k] x^{k-1}
# x P'(x) = sum_{k=1}^{12} k P_c[k] x^k
# -x^2 P(x) = - sum_{k=0}^{12} P_c[k] x^{k+2} = - sum_{j=2}^{14} P_c[j-2] x^j

# Q(x) = P(x) + x P'(x) - x^2 P(x)
# Coeff of x^j in Q(x):
# From P(x): P_c[j] (if 0<=j<=12)
# From x P'(x): j * P_c[j] (if 1<=j<=12)
# From -x^2 P(x): -P_c[j-2] (if 2<=j<=14, i.e., 0<=j-2<=12)

# Q_c has length 15 (indices 0..14)
T = np.zeros((15, 13))
for j in range(15):
    # Contribution from P(x)
    if 0 <= j <= 12:
        T[j, j] += 1.0
    # Contribution from x P'(x)
    if 1 <= j <= 12:
        T[j, j] += j
    # Contribution from -x^2 P(x)
    if 2 <= j <= 14:
        T[j, j-2] += -1.0

# So Q_c = T @ P_c = T @ M @ [c0, c4, c8]
TM = T @ M

def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(F) for given coefficients [c0, c4, c8].
    Uses exact root finding for the critical points of h(x) = x*g(x).
    """
    c0, c4, c8 = coeffs
    
    # Calculate Q(x) coefficients
    Q_c = TM @ np.array([c0, c4, c8])
    
    # Find roots of Q(x) = 0
    # np.roots expects descending order
    Q_desc = Q_c[::-1]
    # Remove trailing zeros to avoid numerical issues, though T should be full rank for relevant coeffs
    # But we must handle the case where the polynomial degree drops.
    # np.roots handles leading zeros by trimming? No, it treats them as coefficients.
    # It's safer to trim leading zeros from Q_desc (which are trailing zeros of Q_c)
    # Actually, np.roots takes [a_n, ..., a_0]. If a_n is 0, it's fine, but efficiency drops.
    
    try:
        roots = np.roots(Q_desc)
    except Exception:
        return 1e10

    # Keep only real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    if len(real_roots) == 0:
        # Fallback: should not happen for real coefficients, but just in case
        return 1e10

    # Evaluate P(x) at the real roots
    # P_c = M @ coeffs
    P_c = M @ np.array([c0, c4, c8])
    P_desc = P_c[::-1]
    
    # Use Horner's method or np.polyval
    # np.polyval expects descending order
    P_vals = np.polyval(P_desc, real_roots)
    
    # Calculate h(x) = x * P(x) * exp(-x^2/2)
    x_vals = real_roots
    g_vals = P_vals * np.exp(-x_vals**2 / 2.0)
    h_vals = x_vals * g_vals
    
    max_val = np.max(np.abs(h_vals))
    
    # The bound is sqrt(2*pi) * max_val^2
    # Wait, the task says "score is the upper bound on C4". 
    # In the inspirations, they used sqrt(2*pi) * Af^2.
    # Let's stick to the definition in Inspiration 1/2 which seems to be the "upper bound on C4" or similar metric.
    # Actually, the prompt says "score is the upper bound on C4".
    # If the seed program returns coefficients, the evaluator computes the score.
    # The "score" in the evaluator is likely the value of the bound.
    # Lower is better.
    
    return math.sqrt(2 * math.pi) * max_val * max_val

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308642])
    best_val = _compute_bound(x0)
    best_x = x0.copy()
    
    # Define bounds for Differential Evolution
    # c0 is negative, c4 positive, c8 small positive
    bounds = [(-5.0, 0.0), (-0.5, 0.5), (-0.01, 0.01)]

    # 1. Differential Evolution for global search
    # Since the objective is fast, we can use a larger population and more iterations.
    try:
        if time.time() < deadline - 30:
            de_result = differential_evolution(
                _compute_bound, bounds, 
                maxiter=200, 
                popsize=20, 
                seed=42,
                tol=1e-10,
                init='sobol',
                polish=False # We do our own polish
            )
            de_val = _compute_bound(de_result.x)
            if de_val < best_val:
                best_val = de_val
                best_x = de_result.x.copy()
    except Exception:
        pass

    # 2. Basin Hopping for local refinement
    # Use Nelder-Mead as the local minimizer because the objective is non-smooth (max norm)
    # L-BFGS-B would struggle with the kinks.
    try:
        if time.time() < deadline - 15:
            def local_minimizer(x):
                res = minimize(_compute_bound, x, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                return res.x, res.fun, {}
            
            bh_result = basinhopping(
                _compute_bound, best_x, 
                minimizer=local_minimizer,
                niter=100, 
                T=0.1, 
                stepsize=0.05,
                seed=42
            )
            bh_val = _compute_bound(bh_result.x)
            if bh_val < best_val:
                best_val = bh_val
                best_x = bh_result.x.copy()
    except Exception:
        pass

    # 3. Final Polish
    # Try a few different local methods from the best point found
    if time.time() < deadline - 5:
        # Nelder-Mead high precision
        try:
            res_nm = minimize(_compute_bound, best_x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            nm_val = _compute_bound(res_nm.x)
            if nm_val < best_val:
                best_val = nm_val
                best_x = res_nm.x.copy()
        except Exception:
            pass
        
        # Powell method (derivative-free, good for non-smooth)
        try:
            res_pow = minimize(_compute_bound, best_x, method='Powell', 
                          options={'maxiter': 500, 'xtol': 1e-14, 'ftol': 1e-16})
            pow_val = _compute_bound(res_pow.x)
            if pow_val < best_val:
                best_val = pow_val
                best_x = res_pow.x.copy()
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
