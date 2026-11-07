"""Rewritten program: local search around the seed to minimize the value."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c0, c1, ..., cn] where c_i is coeff of x^i."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H_nm2 = [1.0]  # H_0
    H_nm1 = [0.0, 2.0] # H_1
    
    for k in range(2, n + 1):
        # H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)
        term1 = [0.0] + [2.0 * c for c in H_nm1]
        term2 = [-2.0 * (k - 1) * c for c in H_nm2]
        
        max_len = max(len(term1), len(term2))
        while len(term1) < max_len:
            term1.append(0.0)
        while len(term2) < max_len:
            term2.append(0.0)
            
        H_n = [t1 + t2 for t1, t2 in zip(term1, term2)]
        
        H_nm2 = H_nm1
        H_nm1 = H_n
        
    return H_nm1

# Precompute Hermite polynomial coefficients
H0_coeffs = np.array(hermite_coeffs(0))
H4_coeffs = np.array(hermite_coeffs(4))
H8_coeffs = np.array(hermite_coeffs(8))
H12_coeffs = np.array(hermite_coeffs(12))

# Values at 0
H0_0 = H0_coeffs[0]
H4_0 = H4_coeffs[0]
H8_0 = H8_coeffs[0]
H12_0 = H12_coeffs[0]

# Since all are even polynomials, g(x) = x^2 * q(x) where q has degree 10 in x
# q(x) = sum_{k=0}^{5} qk * x^{2k}
# We can work in z = x^2: q_z(z) = sum_{k=0}^{5} qk * z^k

# Build the linear mapping from (c0, c1, c2, c3) to q coefficients
# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# Each H_{2m}(x) = x^2 * h_{2m}(x) where h_{2m} is even degree 2m-2
# So q(x) = c0*h0(x) + c1*h4(x) + c2*h8(x) + c3*h12(x)

# Extract the "reduced" polynomials: H_{2m}(x)/x^2 for m=0,2,4,6
# H0(x) = 1, so H0/x^2 = 1/x^2 -- this is problematic!
# Wait, H0 = 1, so c0*H0 contributes c0 to g(x), but g(0)=0 forces c3 to cancel it.
# Let me just compute q directly.

def get_q_coeffs(c0, c1, c2):
    """Return coefficients [q0, q1, ..., q10] of q(x) = g(x)/x^2."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g_coeffs of degree 12
    g_coeffs = np.zeros(13)
    g_coeffs[:len(H0_coeffs)] += c0 * H0_coeffs
    g_coeffs[:len(H4_coeffs)] += c1 * H4_coeffs
    g_coeffs[:len(H8_coeffs)] += c2 * H8_coeffs
    g_coeffs[:len(H12_coeffs)] += c3 * H12_coeffs
    
    # Flip if leading coeff negative
    if g_coeffs[12] < 0:
        g_coeffs = -g_coeffs
    
    # q_coeffs = g_coeffs[2:] (degrees 2 through 12 -> q degrees 0 through 10)
    q_coeffs = g_coeffs[2:]  # length 11
    return q_coeffs

def evaluate_value(c0, c1, c2):
    """
    Compute the value for given coefficients c0, c1, c2.
    Returns the value (float) or None if invalid.
    """
    try:
        c0_f = float(c0)
        c1_f = float(c1)
        c2_f = float(c2)
    except:
        return None

    q_coeffs = get_q_coeffs(c0_f, c1_f, c2_f)
    
    # q(x) = sum q_coeffs[i] * x^i for i=0..10
    # Since g is even, q is even: only even powers survive
    # q(x) = q0 + q2*x^2 + q4*x^4 + ... + q10*x^10
    # Let z = x^2, then q_z(z) = q0 + q2*z + q4*z^2 + ... + q10*z^5
    
    # Extract even-indexed coefficients for z-polynomial
    z_coeffs = q_coeffs[0::2]  # q0, q2, q4, q6, q8, q10 -> degree 5 in z
    
    # Find roots of z-polynomial
    # np.roots expects highest power first
    z_reversed = z_coeffs[::-1]
    
    try:
        z_roots = np.roots(z_reversed)
    except:
        return None
    
    # Get positive real roots
    positive_real_roots = []
    for zr in z_roots:
        if abs(zr.imag) < 1e-10 and zr.real > 1e-12:
            positive_real_roots.append(zr.real)
    
    if not positive_real_roots:
        return None
    
    positive_real_roots.sort(reverse=True)
    
    # For each candidate z-root, check if q changes sign at x = sqrt(z)
    # q(x) = q_z(x^2), so sign of q(x) = sign of q_z(x^2)
    # At x = sqrt(z*), q changes sign iff q_z changes sign at z*
    
    def q_z_val(z):
        return np.polyval(z_coeffs[::-1], z)
    
    r_best = None
    for z_star in positive_real_roots:
        eps = 1e-8 * max(1.0, abs(z_star))
        qz_left = q_z_val(z_star - eps)
        qz_right = q_z_val(z_star + eps)
        
        if qz_left * qz_right < 0:
            # This is a sign-changing root
            r = np.sqrt(z_star)
            r_best = r
            break
    
    if r_best is None:
        return None
    
    value = (r_best ** 2) / (2 * np.pi)
    return value

def construct():
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    if best_val is None:
        return best_c

    start_time = time.time()
    deadline = 200.0  # Safety margin for 210s limit

    if HAS_SCIPY:
        def objective(x):
            val = evaluate_value(x[0], x[1], x[2])
            if val is None:
                return 1e10
            return val
        
        # Multi-start Nelder-Mead with more diverse starts
        rng = np.random.default_rng(42)
        starts = [best_c.copy()]
        
        # Add structured perturbations at multiple scales
        for scale in [0.05, 0.02, 0.01, 0.005]:
            for _ in range(3):
                perturbation = rng.normal(0, scale, size=3)
                starts.append(best_c + perturbation)
        
        # Also try some larger jumps
        for _ in range(5):
            perturbation = rng.normal(0, 0.1, size=3)
            starts.append(best_c + perturbation)
        
        for x0 in starts:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                if time.time() > deadline - 5:
                    break
                val_new = evaluate_value(res.x[0], res.x[1], res.x[2])
                if val_new is not None and val_new < best_val:
                    best_c = res.x.copy()
                    best_val = val_new
            except:
                pass
        
        # Final high-precision polish
        if time.time() < deadline - 15:
            try:
                res_polish = minimize(objective, best_c, method='Nelder-Mead', 
                                      options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
                val_polish = evaluate_value(res_polish.x[0], res_polish.x[1], res_polish.x[2])
                if val_polish is not None and val_polish < best_val:
                    best_c = res_polish.x.copy()
                    best_val = val_polish
            except:
                pass
        
        # Additional local refinement with even tighter settings
        if time.time() < deadline - 10:
            try:
                res_final = minimize(objective, best_c, method='Nelder-Mead', 
                                     options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16})
                val_final = evaluate_value(res_final.x[0], res_final.x[1], res_final.x[2])
                if val_final is not None and val_final < best_val:
                    best_c = res_final.x.copy()
                    best_val = val_final
            except:
                pass

    return best_c
# EVOLVE-BLOCK-END
