"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """Find optimal coefficients c0, c1, c2 to minimize the uncertainty constant."""
    
    # We parameterize the polynomial Q(u) = g(x)/x^2 using c1 and c2.
    # We fix c3 = 1 to fix the scale, as the roots are invariant to scaling.
    # c0 is determined by the condition g(0)=0, but it doesn't appear in Q(u) coefficients
    # except to ensure the constant term of g is 0, which is already handled by the structure.
    # Actually, Q(u) coefficients depend only on c1, c2, c3.
    # Let's verify:
    # g(x) = c0*1 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - ... + 1680) + c3*(4096x^12 - ... + 4435200)
    # g(0) = c0 + 12c1 + 1680c2 + 4435200c3 = 0  => c0 = -12c1 - 1680c2 - 4435200c3
    # Q(u) = g(sqrt(u))/u
    # g(sqrt(u)) = c0 + c1(16u^2 - 48u + 12) + c2(256u^4 - 3072u^3 + 11520u^2 - 13440u + 1680) + c3(4096u^6 - 98304u^5 + 1032192u^4 - 5308416u^3 + 12096000u^2 - 12902400u + 4435200)
    # The constant term of g(sqrt(u)) is c0 + 12c1 + 1680c2 + 4435200c3 = 0.
    # So g(sqrt(u)) is divisible by u.
    # Q(u) = c1(16u - 48) + c2(256u^3 - 3072u^2 + 11520u - 13440) + c3(4096u^5 - 98304u^4 + 1032192u^3 - 5308416u^2 + 12096000u - 12902400)
    
    # Let c3 = 1.
    # Q(u) = 4096u^5 - 98304u^4 + (1032192 + 256c2)u^3 + (-5308416 - 3072c2)u^2 + (12096000 + 11520c2 - 48c1)u + (-12902400 - 13440c2 + 16c1)
    # Wait, let's re-calculate the coefficients carefully.
    # Coeff of u^5: c3 * 4096 = 4096
    # Coeff of u^4: c3 * (-98304) = -98304
    # Coeff of u^3: c3 * 1032192 + c2 * 256 = 1032192 + 256c2
    # Coeff of u^2: c3 * (-5308416) + c2 * (-3072) = -5308416 - 3072c2
    # Coeff of u^1: c3 * 12096000 + c2 * 11520 + c1 * 16 = 12096000 + 11520c2 + 16c1
    # Coeff of u^0: c3 * (-12902400) + c2 * (-13440) + c1 * (-48) = -12902400 - 13440c2 - 48c1
    
    # Let p = c2, q = c1.
    def get_Q_coeffs(p, q):
        a5 = 4096.0
        a4 = -98304.0
        a3 = 1032192.0 + 256.0 * p
        a2 = -5308416.0 - 3072.0 * p
        a1 = 12096000.0 + 11520.0 * p + 16.0 * q
        a0 = -12902400.0 - 13440.0 * p - 48.0 * q
        return np.array([a5, a4, a3, a2, a1, a0])

    def max_sign_change_root(coeffs):
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        positive_roots = real_roots[real_roots > 1e-9]
        if len(positive_roots) == 0:
            return 0.0
        
        positive_roots.sort(reverse=True)
        
        for r in positive_roots:
            # Check sign change
            # Use a small epsilon relative to r
            eps = 1e-7 * max(1.0, abs(r))
            Q_left = np.polyval(coeffs, r - eps)
            Q_right = np.polyval(coeffs, r + eps)
            
            # If signs are opposite, it's a sign change
            if Q_left * Q_right < 0:
                return r
        return 0.0

    def objective(params):
        p, q = params
        try:
            coeffs = get_Q_coeffs(p, q)
            r = max_sign_change_root(coeffs)
            # We want to maximize r, so minimize -r
            if r == 0.0:
                return 1e100
            return -r
        except Exception:
            return 1e100

    # Seed values
    p_seed = 1.0 / 3240.0
    q_seed = 1.0 / 25.0
    
    best_r = 0.0
    best_params = (p_seed, q_seed)
    
    # Multi-start Nelder-Mead
    # We search in a box around the seed
    # The seed is small, so let's look in a range that covers orders of magnitude
    # or just a reasonable physical range.
    # c1 ~ 0.04, c2 ~ 0.0003
    # Let's try ranges: p in [-1, 1], q in [-1, 1]
    
    rng = np.random.default_rng(42)
    
    # Generate starting points
    starts = []
    # Include seed
    starts.append((p_seed, q_seed))
    
    # Add random points in a log-uniform or uniform distribution
    # Uniform in [-1, 1]
    for _ in range(50):
        p = rng.uniform(-1.0, 1.0)
        q = rng.uniform(-1.0, 1.0)
        starts.append((p, q))
        
    # Also try some scaled versions
    for scale in [0.1, 0.01, 10.0]:
        for _ in range(10):
            p = rng.uniform(-scale, scale)
            q = rng.uniform(-scale, scale)
            starts.append((p, q))

    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < -best_r: # Since we minimize -r, lower fun is better (more negative)
                best_r = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Recover c0, c1, c2
    p_opt, q_opt = best_params
    c2 = p_opt
    c1 = q_opt
    c3 = 1.0
    c0 = -12.0 * c1 - 1680.0 * c2 - 4435200.0 * c3
    
    # Check if the result is valid (r > 0)
    if best_r <= 0:
        # Fallback to seed
        c2 = 1.0 / 3240.0
        c1 = 1.0 / 25.0
        c3 = 1.0
        c0 = -12.0 * c1 - 1680.0 * c2 - 4435200.0 * c3

    return [c0, c1, c2]

# EVOLVE-BLOCK-END
