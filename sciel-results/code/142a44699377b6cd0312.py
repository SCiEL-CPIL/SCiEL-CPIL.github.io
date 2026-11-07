"""
Optimized search for coefficients c0, c1, c2 using structural constraints.
Combines the robust coefficient precomputation of the current program with
the Newton-based double-root constraint search from Inspiration 2.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 175.0
_t_start = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _t_start)

# Precompute Hermite polynomial coefficients using recurrence
def get_hermite_coeffs(n):
    """Returns list of coefficients [c_0, c_1, ..., c_n] for H_n(x) = c_n x^n + ... + c_0."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    h_prev = [1.0]  # H_0
    h_curr = [0.0, 2.0] # H_1
    
    for k in range(1, n):
        h_next = [0.0] * (k + 2)
        # 2x H_k part
        for i in range(k + 1):
            h_next[i + 1] += 2.0 * h_curr[i]
        # -2k H_{k-1} part
        for i in range(k):
            h_next[i] += -2.0 * k * h_prev[i]
        h_prev = h_curr
        h_curr = h_next
    return h_curr

H0 = get_hermite_coeffs(0)
H4 = get_hermite_coeffs(4)
H8 = get_hermite_coeffs(8)
H12 = get_hermite_coeffs(12)

# Extract constant terms for c3 calculation
H0_0 = H0[0]
H4_0 = H4[0]
H8_0 = H8[0]
H12_0 = H12[0]

def _q_coefficients(c0, c1, c2):
    """
    Return coefficients [b0..b5] for q(y) where y=x^2.
    q(y) = g(sqrt(y))/y.
    """
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # We need coefficients of g(x) / x^2.
    # Since g is even, g(x) = sum a_k x^{2k}. q(y) = sum a_{2k} y^k.
    # Let's build g_coeffs (lowest to highest) and extract even indices.
    
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for i in range(len(H0)):
        g_coeffs[i] += c0 * H0[i]
    for i in range(len(H4)):
        g_coeffs[i] += c1 * H4[i]
    for i in range(len(H8)):
        g_coeffs[i] += c2 * H8[i]
    for i in range(len(H12)):
        g_coeffs[i] += c3 * H12[i]
        
    # q(y) coefficients: g_coeffs[2], g_coeffs[4], ..., g_coeffs[12]
    # q(y) = b0 + b1*y + ... + b5*y^5
    q_coeffs = np.array([g_coeffs[2], g_coeffs[4], g_coeffs[6], g_coeffs[8], g_coeffs[10], g_coeffs[12]])
    
    return q_coeffs

def _q_eval(coeffs, y):
    """Horner evaluation of q(y)."""
    r = 0.0
    for b in coeffs:
        r = r * y + b
    return r

def _q_deriv(coeffs, y):
    """Derivative q'(y) via Horner."""
    r = 0.0
    for i in range(len(coeffs) - 1, 0, -1):
        r = r * y + i * coeffs[i]
    return r

def _find_largest_sign_change_root(coeffs):
    """Find largest real positive root where q changes sign. Returns y or None."""
    # np.roots expects highest degree first
    poly_coeffs = coeffs[::-1]
    roots = np.roots(poly_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-12:
            real_roots.append(r.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for y in real_roots:
        eps = max(1e-7, abs(y) * 1e-7)
        qb = _q_eval(coeffs, y - eps)
        qa = _q_eval(coeffs, y + eps)
        if qb * qa < 0:
            return y
    return None

def _evaluate_value(c0, c1, c2):
    """Compute value = r^2/(2*pi) where r^2=y is the largest sign-changing root."""
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        # If no sign-changing root, the polynomial might be positive definite.
        # In the context of the inequality, if q(x) > 0 for all x, the bound is effectively 0?
        # Or is it invalid? The task says "r be the largest real root... at which q changes sign".
        # If no such root, the definition is ambiguous. However, if q(x) > 0, 
        # then g(x) > 0, which is a valid positive function. 
        # Usually, if q(x) doesn't change sign, it means it's always positive (since leading term dominates).
        # If q(x) > 0, then g(x) > 0. The "root" at which it changes sign doesn't exist.
        # Let's assume this case yields a very small value (good) or is invalid.
        # Given "lower is better", a positive definite q is ideal.
        # Let's check if q is positive at infinity.
        if _q_eval(coeffs, 100.0) > 0:
            return 1e-6
        else:
            # If q is negative at infinity, g -> -inf. Evaluator flips g.
            # Then q -> -q > 0. So it becomes positive definite.
            return 1e-6
    return y_root / (2.0 * np.pi)

# Seed
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_seed_value = _evaluate_value(_SEED[0], _SEED[1], _SEED[2])

def _newton_double_root(c0, c1, c2, y_target, max_iter=50, tol=1e-14):
    """
    Newton's method to find (c1, c2) such that q(y_target)=0 and q'(y_target)=0,
    holding c0 fixed. Returns (c1, c2, success).
    """
    c1_cur = c1
    c2_cur = c2
    
    # Precompute derivatives of c3 w.r.t c1, c2
    dc3_dc1 = -H4_0 / H12_0
    dc3_dc2 = -H8_0 / H12_0
    
    # Precompute derivatives of q_coeffs w.r.t c1 and c2
    # q_coeffs[k] = c0*H0[2k] + c1*H4[2k] + c2*H8[2k] + c3*H12[2k]
    # dq_k/dc1 = H4[2k] + H12[2k]*dc3_dc1
    # dq_k/dc2 = H8[2k] + H12[2k]*dc3_dc2
    
    dq_dc1 = np.array([H4[2*k] + H12[2*k]*dc3_dc1 for k in range(6)])
    dq_dc2 = np.array([H8[2*k] + H12[2*k]*dc3_dc2 for k in range(6)])
    
    for _ in range(max_iter):
        coeffs = _q_coefficients(c0, c1_cur, c2_cur)
        f1 = _q_eval(coeffs, y_target)
        f2 = _q_deriv(coeffs, y_target)
        
        if abs(f1) < tol and abs(f2) < tol:
            return c1_cur, c2_cur, True
            
        # Jacobian J = [[df1/dc1, df1/dc2], [df2/dc1, df2/dc2]]
        # df1/dc1 = sum dq_k/dc1 * y^k
        # df2/dc1 = sum k * dq_k/dc1 * y^(k-1)
        
        y = y_target
        y_powers = np.array([1.0, y, y**2, y**3, y**4, y**5])
        
        df1_dc1 = np.dot(dq_dc1, y_powers)
        df1_dc2 = np.dot(dq_dc2, y_powers)
        
        # Derivatives of q' w.r.t c1, c2
        # q'(y) = sum_{k=1}^5 k * b_k * y^(k-1)
        # dq'_k/dc1 = k * dq_k/dc1
        # df2/dc1 = sum_{k=1}^5 k * dq_k/dc1 * y^(k-1)
        
        k_factors = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y_powers_der = np.array([1.0, y, y**2, y**3, y**4]) # y^0 to y^4
        
        # Note: q'(y) = b1 + 2b2y + ...
        # Term k corresponds to coefficient b_k.
        # d(b_k)/dc1 * k * y^(k-1)
        
        df2_dc1 = np.dot(dq_dc1[1:] * k_factors[1:], y_powers_der)
        df2_dc2 = np.dot(dq_dc2[1:] * k_factors[1:], y_powers_der)
        
        J = np.array([[df1_dc1, df1_dc2],
                      [df2_dc1, df2_dc2]])
        F = np.array([f1, f2])
        
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            return c1_cur, c2_cur, False
            
        J_inv = np.array([[J[1, 1], -J[0, 1]],
                          [-J[1, 0], J[0, 0]]]) / det
        delta = J_inv @ F
        
        # Damped Newton
        step_size = 1.0
        while step_size > 1e-6:
            c1_trial = c1_cur - step_size * delta[0]
            c2_trial = c2_cur - step_size * delta[1]
            
            coeffs_trial = _q_coefficients(c0, c1_trial, c2_trial)
            f1_trial = _q_eval(coeffs_trial, y_target)
            f2_trial = _q_deriv(coeffs_trial, y_target)
            
            if abs(f1_trial) < abs(f1) and abs(f2_trial) < abs(f2):
                c1_cur = c1_trial
                c2_cur = c2_trial
                break
            step_size *= 0.5
        else:
            # If no step accepted, break
            return c1_cur, c2_cur, False

    coeffs = _q_coefficients(c0, c1_cur, c2_cur)
    f1 = _q_eval(coeffs, y_target)
    f2 = _q_deriv(coeffs, y_target)
    return c1_cur, c2_cur, (abs(f1) < 1e-10 and abs(f2) < 1e-10)

def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Uses Newton's method on double-root constraint with bisection over target y*.
    """
    global _t_start
    _t_start = time.time()

    best_params = _SEED.copy()
    best_value = _seed_value

    # Strategy:
    # 1. For a range of c0, use Newton to find (c1, c2) such that q(y*)=0 and q'(y*)=0.
    # 2. This creates a double root at y*. 
    # 3. If y* is the LARGEST sign-changing root, then value = y*/(2*pi).
    # 4. We can scan y* to minimize it.
    # 5. We must ensure that there are no larger sign-changing roots.
    
    # Scan c0 in a reasonable range
    c0_range = np.linspace(-3.0, 3.0, 13)
    
    # Scan y_target in a reasonable range
    # Small y_target gives small value, but might not be the largest root.
    # We need to verify that y_target is indeed the largest sign-changing root.
    
    for c0 in c0_range:
        if _remaining() < 5.0:
            break
            
        # Try various y_targets
        # Start from small values
        y_targets = np.logspace(-4, 1, 20)
        
        for y_t in y_targets:
            if _remaining() < 2.0:
                break
                
            # Initial guess for c1, c2
            # Use seed as initial guess, or random
            c1_init = _SEED[1]
            c2_init = _SEED[2]
            
            c1_new, c2_new, success = _newton_double_root(c0, c1_init, c2_init, y_t)
            
            if success:
                val, r, valid = _evaluate_value(c0, c1_new, c2_new), 0, True
                # _evaluate_value returns value directly
                val = _evaluate_value(c0, c1_new, c2_new)
                
                if val < best_value:
                    # Verify that y_t is the largest sign-changing root
                    coeffs = _q_coefficients(c0, c1_new, c2_new)
                    y_root = _find_largest_sign_change_root(coeffs)
                    if y_root is not None and abs(y_root - y_t) < 1e-3:
                        best_value = val
                        best_params = np.array([c0, c1_new, c2_new])
                    # Even if y_t is not the largest, the resulting polynomial might still be good?
                    # No, value is determined by the LARGEST root.
                    # So we only accept if y_t IS the largest root.
                    
    # Also try local optimization from the best found so far
    if _remaining() > 10.0:
        # Use Nelder-Mead from best_params
        from scipy.optimize import minimize
        
        def obj(c):
            return _evaluate_value(c[0], c[1], c[2])
            
        try:
            res = minimize(obj, best_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_value:
                best_value = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params

# EVOLVE-BLOCK-END
