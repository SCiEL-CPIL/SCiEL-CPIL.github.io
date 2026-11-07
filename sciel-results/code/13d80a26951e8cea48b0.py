# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

# Precompute the Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Coefficients for each Hermite polynomial (constant term, x^2 term, x^4 term, ...)
# H0: [1]
# H4: [12, -48, 16]  (coeffs of x^0, x^2, x^4)
# H8: [168, -1344, 2688, -1344, 256]  (coeffs of x^0, x^2, x^4, x^6, x^8)
# H12: [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]  (coeffs of x^0, x^2, x^4, x^6, x^8, x^10, x^12)

H0_coeffs = [1.0]
H4_coeffs = [12.0, -48.0, 16.0]
H8_coeffs = [168.0, -1344.0, 2688.0, -1344.0, 256.0]
H12_coeffs = [18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0]

# H12(0) = 18480
H12_0 = 18480.0

def compute_q_coefficients(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2 where
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    
    q(x) is an even polynomial of degree 10, so we work with y = x^2.
    q(x) = sum_{k=0}^{5} q_k * y^k where y = x^2
    
    Returns: list of coefficients [q_0, q_1, ..., q_5] for powers y^0 to y^5
    """
    # c3 is determined by g(0) = 0
    H0_0 = H0_coeffs[0]  # 1
    H4_0 = H4_coeffs[0]  # 12
    H8_0 = H8_coeffs[0]  # 168
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Now compute g(x) coefficients. g(x) is even, degree 12.
    # g(x) = sum_{k=0}^{6} g_k * x^{2k}
    # g(x)/x^2 = sum_{k=0}^{5} g_{k+1} * x^{2k} = sum_{k=0}^{5} q_k * y^k
    
    # Initialize g coefficients (length 7, for x^0 to x^12)
    g_coeffs = [0.0] * 7
    
    # Add c0 * H0
    for i, coeff in enumerate(H0_coeffs):
        g_coeffs[i] += c0 * coeff
    
    # Add c1 * H4
    for i, coeff in enumerate(H4_coeffs):
        g_coeffs[i] += c1 * coeff
    
    # Add c2 * H8
    for i, coeff in enumerate(H8_coeffs):
        g_coeffs[i] += c2 * coeff
    
    # Add c3 * H12
    for i, coeff in enumerate(H12_coeffs):
        g_coeffs[i] += c3 * coeff
    
    # q(x) = g(x)/x^2, so q_k = g_{k+1} for k = 0..5
    q_coeffs = [g_coeffs[k+1] for k in range(6)]
    
    return q_coeffs

def evaluate_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real root of q
    at which q changes sign.
    
    Returns the value, or float('inf') if no valid root exists.
    """
    q_coeffs = compute_q_coefficients(c0, c1, c2)
    
    # q(y) = sum_{k=0}^{5} q_coeffs[k] * y^k where y = x^2 >= 0
    # We need to find roots of q(y) = 0 for y >= 0 where q changes sign.
    
    # Use numpy to find roots of the polynomial in y
    # q_coeffs is [q_0, q_1, ..., q_5] for y^0 to y^5
    # numpy.roots expects coefficients in descending order: [q_5, q_4, ..., q_0]
    poly_coeffs_desc = list(reversed(q_coeffs))
    
    try:
        roots_y = np.roots(poly_coeffs_desc)
    except Exception:
        return float('inf')
    
    # Filter for real, non-negative roots
    valid_roots = []
    for r in roots_y:
        if np.isreal(r):
            r_real = np.real(r)
            if r_real >= -1e-10:  # allow small negative due to numerical error
                r_real = max(0.0, r_real)
                valid_roots.append(r_real)
    
    if not valid_roots:
        return float('inf')
    
    # For each valid root, check if q changes sign there.
    # A sign change means the root has odd multiplicity.
    # We can check by evaluating q at points slightly below and above the root.
    
    def q_eval(y):
        """Evaluate q(y) = sum q_coeffs[k] * y^k"""
        result = 0.0
        for k, coeff in enumerate(q_coeffs):
            result += coeff * (y ** k)
        return result
    
    sign_change_roots = []
    for r in valid_roots:
        if r < 1e-12:
            # Root at y=0, check sign change from positive side
            eps = 1e-8
            val_before = q_eval(0.0)  # q(0) = q_coeffs[0]
            val_after = q_eval(eps)
            if val_before * val_after < 0:
                sign_change_roots.append(r)
        else:
            eps = 1e-8 * max(1.0, r)
            val_before = q_eval(r - eps)
            val_after = q_eval(r + eps)
            if val_before * val_after < 0:
                sign_change_roots.append(r)
    
    if not sign_change_roots:
        return float('inf')
    
    # r is the largest real root at which q changes sign
    # Note: r in the problem is x, and y = x^2, so the root in x is sqrt(y)
    # Wait, let me re-read the problem.
    # "Let q = g / x^2, which must divide exactly. Let r be the largest real root of q at which q changes sign."
    # q is a function of x. q(x) = g(x)/x^2. So q is an even function of x.
    # The roots of q are the same as the roots of g except possibly at x=0.
    # Since g(0)=0 and g is even, g(x) = x^2 * q(x).
    # The roots of q(x) = 0 are the roots of g(x) = 0 excluding x=0 (if it's a simple root of g).
    # q(x) is even, so its roots come in pairs ±r.
    # "r be the largest real root" - this likely means the largest positive root.
    
    # So r is the largest positive x such that q(x) = 0 and q changes sign at x.
    # In terms of y = x^2, the root is y = r^2.
    
    # So we need the largest y in sign_change_roots, and r = sqrt(y).
    
    max_y = max(sign_change_roots)
    r = np.sqrt(max_y)
    
    value = r**2 / (2 * np.pi)
    
    return value

def construct():
    """
    Search for coefficients [c0, c1, c2] that minimize the value.
    Strategy:
    1. Parameterize coefficients such that g(x) is non-negative (Sum of Squares structure).
       Specifically, we look for g(x) = (a*x^2 + b)^2 * (c*x^2 + d) ... or similar.
       Actually, a simpler robust strategy for "non-negative polynomial with roots":
       We want g(x) >= 0 for all x. This implies q(y) >= 0 for all y >= 0.
       If q(y) has real positive roots, they must be of even multiplicity for q to stay non-negative.
       However, the problem asks for the largest root where q CHANGES SIGN.
       If q(y) >= 0 for all y, it never changes sign at a root (it touches 0).
       Wait. "If g tends to a negative limit at infinity, g is replaced by -g."
       This implies the evaluator ensures the leading coefficient is positive.
       If the leading coefficient is positive and g(x) -> +inf, and g(0)=0,
       does g have to cross zero?
       If g(x) >= 0 for all x, then the only roots are where g=0.
       At a root of even multiplicity, g does not change sign.
       At a root of odd multiplicity, g changes sign.
       
       If g changes sign at r, then g is negative on one side and positive on the other.
       Since g(0)=0 and g is even, g(x) has the same sign for small +/- x.
       If g(x) < 0 for small x != 0, then g must cross 0 to become positive at infinity.
       So r is the largest crossing.
       
       To minimize r, we want the crossing to happen as early as possible?
       No, value = r^2 / (2*pi). Lower is better.
       So we want the LARGEST sign-changing root to be as SMALL as possible.
       
       This suggests we want g(x) to stay negative (or close to 0) for as long as possible,
       or rather, we want the first positive crossing to be small?
       No, "largest real root".
       If there are multiple sign-changing roots, we take the largest one.
       To minimize the largest sign-changing root, we should ideally have NO sign-changing roots larger than some small R.
       Ideally, we want g(x) to be negative everywhere except near 0?
       But g(0)=0. If g(x) < 0 for all x != 0, then there are no sign-changing roots (it stays negative).
       But g(x) -> +inf (due to -g replacement if needed, or we choose positive leading coeff).
       If g(x) -> +inf and g(x) < 0 for small x, it MUST cross zero.
       So there is at least one sign-changing root.
       We want the LARGEST one to be small.
       This implies we want ALL sign-changing roots to be small.
       Ideally, g(x) becomes positive very quickly and stays positive.
       So g(x) < 0 for x in (0, R) and g(x) > 0 for x > R.
       Then the only sign-changing roots are at R (and maybe 0 if it crosses, but g(0)=0).
       If g(x) = x^2 (x^2 - R^2) ... g(0)=0.
       For x < R, g < 0. For x > R, g > 0.
       Sign change at R.
       Largest sign changing root is R.
       Value = R^2 / (2*pi).
       We want to minimize R.
       
       So the goal is to find a polynomial g in the span {H0, H4, H8, H12} such that:
       1. Leading coeff > 0.
       2. g(0) = 0.
       3. g(x) < 0 for 0 < x < R.
       4. g(x) > 0 for x > R.
       5. R is minimized.
       
       This is equivalent to finding the smallest R such that there exists a combination
       where g(x) has a double root at R (to just touch 0? No, it must cross).
       Actually, if it crosses at R, g(R)=0 and g'(R)=0 is not required, but for the "largest" root to be minimal,
       we likely want the polynomial to be "flat" or constrained.
       
       Let's use the SDP/Convex idea:
       Minimize R^2 subject to g(x) >= 0 for x >= R and g(x) <= 0 for 0 <= x <= R?
       This is non-convex.
       
       Alternative:
       We can search for (c0, c1, c2) that minimizes the largest positive root.
       We can use the fact that if we scale the coefficients, the roots don't change?
       No, c3 is fixed by c0, c1, c2.
       g(x) is linear in c0, c1, c2.
       Let g(x) = c0 * (H0 + k0*H12) + c1 * (H4 + k1*H12) + c2 * (H8 + k2*H12).
       Let P0 = H0 + (H0(0)/H12(0)) H12
       Let P1 = H4 + (H4(0)/H12(0)) H12
       Let P2 = H8 + (H8(0)/H12(0)) H12
       Then g(x) = c0 P0(x) + c1 P1(x) + c2 P2(x).
       Note that P0(0)=P1(0)=P2(0)=0.
       
       We want to find c0, c1, c2 such that the largest sign-changing root of g is minimized.
       
       Let's try a specific structural search:
       Assume g(x) has a double root at r? No, sign change implies odd multiplicity.
       If we want the LARGEST sign-changing root to be small, we want g to turn positive early.
       
       Let's try to optimize (c0, c1, c2) using Nelder-Mead on the value function directly,
       but with a better initialization and handling of the "no sign change" case.
       
       The previous program scored 0.128.
       The seed is [-1.13, 0.04, 0.000308].
       
       Let's try a grid search or random restarts with Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    # Precompute basis polynomials P0, P1, P2
    # P_i(x) = H_i(x) - (H_i(0)/H12(0)) H12(x)
    # Wait, c3 = -(c0 H0(0) + c1 H4(0) + c2 H8(0)) / H12(0)
    # g = c0 H0 + c1 H4 + c2 H8 + c3 H12
    # g = c0 (H0 - H0(0)/H12(0) H12) + c1 (H4 - H4(0)/H12(0) H12) + c2 (H8 - H8(0)/H12(0) H12)
    
    k0 = H0_coeffs[0] / H12_0
    k1 = H4_coeffs[0] / H12_0
    k2 = H8_coeffs[0] / H12_0
    
    # P0 = H0 - k0 H12
    # P1 = H4 - k1 H12
    # P2 = H8 - k2 H12
    
    def get_q_from_basis(c0, c1, c2):
        # g = c0 P0 + c1 P1 + c2 P2
        # We need coefficients of g.
        # H0, H4, H8, H12 coeffs are known.
        # g_coeffs = c0 * (H0_coeffs - k0 * H12_coeffs) + ...
        # But H0, H4, H8 have different lengths. Pad with zeros.
        
        g_coeffs = [0.0] * 7
        for i in range(len(H0_coeffs)):
            g_coeffs[i] += c0 * (H0_coeffs[i] - (k0 * H12_coeffs[i] if i < len(H12_coeffs) else 0))
        for i in range(len(H4_coeffs)):
            g_coeffs[i] += c1 * (H4_coeffs[i] - (k1 * H12_coeffs[i] if i < len(H12_coeffs) else 0))
        for i in range(len(H8_coeffs)):
            g_coeffs[i] += c2 * (H8_coeffs[i] - (k2 * H12_coeffs[i] if i < len(H12_coeffs) else 0))
            
        # q_coeffs = g_coeffs[1:]
        return g_coeffs[1:]

    def value_from_q(q_coeffs):
        # q_coeffs are for y^0 to y^5
        # Find roots of q(y)
        if not any(abs(c) > 1e-12 for c in q_coeffs):
            return float('inf')
            
        poly_desc = list(reversed(q_coeffs))
        try:
            roots = np.roots(poly_desc)
        except Exception:
            return float('inf')
            
        valid_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9:
                r_real = r.real
                if r_real >= -1e-9:
                    valid_roots.append(max(0.0, r_real))
                    
        if not valid_roots:
            return float('inf')
            
        # Check sign change
        def q_eval(y):
            res = 0.0
            for k, c in enumerate(q_coeffs):
                res += c * (y ** k)
            return res
            
        sign_change_roots = []
        for r in valid_roots:
            if r < 1e-9:
                # Check at 0
                # q(0) = q_coeffs[0]
                # q(eps)
                eps = 1e-5
                v0 = q_eval(0)
                v1 = q_eval(eps)
                if v0 * v1 < 0:
                    sign_change_roots.append(r)
            else:
                eps = 1e-5 * max(1.0, r)
                v1 = q_eval(r - eps)
                v2 = q_eval(r + eps)
                if v1 * v2 < 0:
                    sign_change_roots.append(r)
                    
        if not sign_change_roots:
            return float('inf')
            
        max_y = max(sign_change_roots)
        r = np.sqrt(max_y)
        return r**2 / (2 * np.pi)

    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return float('inf')
        try:
            q = get_q_from_basis(x[0], x[1], x[2])
            return value_from_q(q)
        except Exception:
            return float('inf')

    # Initial guesses
    # Seed: [-1.13, 0.04, 0.000308]
    # Try to find a point with small value.
    # Random restarts
    best_val = float('inf')
    best_x = None
    
    rng = np.random.default_rng(42)
    
    # Try seed first
    seed_x = np.array([-1.13, 0.04, 0.000308])
    v = objective(seed_x)
    if v < best_val:
        best_val = v
        best_x = seed_x.copy()
        
    # Random search
    num_restarts = 100
    for i in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        # Random point in a reasonable box
        # c0, c1, c2
        # Scale: H12 is large. c3 will be small.
        # Let's try [-5, 5]
        x0 = rng.uniform(-5, 5, 3)
        
        # Local optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    # If we found a good point, refine it
    if best_x is not None and best_val < float('inf'):
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if best_x is None:
        return [-1.13, 0.04, 0.000308]
        
    return best_x.tolist()
# EVOLVE-BLOCK-END
