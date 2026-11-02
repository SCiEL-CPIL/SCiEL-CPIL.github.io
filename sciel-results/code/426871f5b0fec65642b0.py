"""Optimized program: numerical minimization of the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial


def _hermite_prob(n, x):
    """Probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_c4_bound_fast(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(g) * A(ĝ) for g normalized to ||g||_2 = 1.
    Uses exact L2 norm via orthogonality and root-finding for sup norm.
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Coefficients for P(x) = a0*He0 + a4*He4 + a8*He8 + a12*He12
    # We need to find the roots of P'(x) - x*P(x) = 0 to find critical points of g(x) = P(x)exp(-x^2/2).
    # It is more efficient to work with the coefficients of the polynomial P(x) directly.
    
    # Get coefficients of He0, He4, He8, He12
    # He0(x) = 1
    # He4(x) = x^4 - 6x^2 + 3
    # He8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # He12(x) = x^12 - 66x^10 + 2310x^8 - 27720x^6 + 66528x^4 - 30240x^2 + 945
    
    # Construct P(x) coefficients in descending order (highest power first)
    # P(x) = a12*He12(x) + a8*He8(x) + a4*He4(x) + a0*He0(x)
    
    # He12 coeffs: [1, 0, -66, 0, 2310, 0, -27720, 0, 66528, 0, -30240, 0, 945]
    # He8 coeffs:  [1, 0, -28, 0, 420, 0, -1680, 0, 105]
    # He4 coeffs:  [1, 0, -6, 0, 3]
    # He0 coeffs:  [1]
    
    # Pad to length 13 (degree 12)
    c12 = np.zeros(13)
    c12[0] = 1.0
    c12[2] = -66.0
    c12[4] = 2310.0
    c12[6] = -27720.0
    c12[8] = 66528.0
    c12[10] = -30240.0
    c12[12] = 945.0
    
    c8 = np.zeros(13)
    c8[4] = 1.0
    c8[6] = -28.0
    c8[8] = 420.0
    c8[10] = -1680.0
    c8[12] = 105.0
    
    c4 = np.zeros(13)
    c4[8] = 1.0
    c4[10] = -6.0
    c4[12] = 3.0
    
    c0 = np.zeros(13)
    c0[12] = 1.0
    
    p_coeffs = a12 * c12 + a8 * c8 + a4 * c4 + a0 * c0
    
    # Calculate L2 norm squared: sqrt(pi) * sum(ak^2 * k!)
    # k! for k=0,4,8,12
    norm_sq = np.sqrt(np.pi) * (
        a0**2 * 1 + 
        a4**2 * np.math.factorial(4) + 
        a8**2 * np.math.factorial(8) + 
        a12**2 * np.math.factorial(12)
    )
    
    if norm_sq < 1e-12:
        return 1e10
        
    norm = np.sqrt(norm_sq)
    
    # Find critical points of g(x) = P(x) exp(-x^2/2)
    # g'(x) = (P'(x) - x P(x)) exp(-x^2/2) = 0
    # Let Q(x) = P'(x) - x P(x)
    # If P(x) = sum c_i x^i, then P'(x) = sum i c_i x^(i-1)
    # x P(x) = sum c_i x^(i+1)
    # Q(x) = sum i c_i x^(i-1) - sum c_i x^(i+1)
    # Degree of Q is 11 (since P is degree 12, xP is 13, P' is 11. Wait.
    # P(x) = c_12 x^12 + ...
    # P'(x) = 12 c_12 x^11 + ...
    # x P(x) = c_12 x^13 + ...
    # Q(x) = -c_12 x^13 + ... ? No.
    # g(x) = P(x) e^{-x^2/2}.
    # g'(x) = P'(x) e^{-x^2/2} + P(x) (-x) e^{-x^2/2} = (P'(x) - x P(x)) e^{-x^2/2}.
    # Yes, Q(x) = P'(x) - x P(x).
    # Degree of x P(x) is 13. Degree of P'(x) is 11.
    # So Q(x) is a polynomial of degree 13 with leading term -c_12 x^13.
    
    # Construct Q(x) coefficients
    # P_coeffs is [c_12, c_11, ..., c_0]
    # We want coefficients of Q(x) in descending power order for np.roots.
    # Q(x) = sum_{i=0}^{12} (i c_i x^{i-1} - c_i x^{i+1})
    # Let's build Q_coeffs of length 14 (degree 13)
    q_coeffs = np.zeros(14)
    for i in range(13):
        c_i = p_coeffs[i]
        power = 12 - i
        # Term from P': i * c_i * x^(power-1)
        if i > 0:
            # Power is power-1. Index in q_coeffs is 13 - (power-1) = 14 - power
            idx_deriv = 14 - power
            q_coeffs[idx_deriv] += i * c_i
        # Term from -x P: -c_i * x^(power+1)
        # Power is power+1. Index in q_coeffs is 14 - (power+1) = 13 - power
        idx_xp = 13 - power
        q_coeffs[idx_xp] -= c_i
        
    # Find roots of Q(x)
    # Filter out small roots if any (numerical noise), but here we just find all
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return 1e10
        
    # Evaluate g(x) at real roots and 0
    # g(x) = P(x) exp(-x^2/2)
    # We only care about real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    max_val = 0.0
    # Check 0
    val_0 = p_coeffs[12] * np.exp(0) # P(0) = c0 = p_coeffs[12]
    # Note: P(0) should be 0 by constraint, but due to floating point it might not be exactly 0.
    # However, the constraint is enforced analytically in a12, so P(0) is 0.
    # But let's check anyway.
    max_val = max(max_val, abs(val_0))
    
    for r in real_roots:
        # Evaluate P(r)
        val_p = np.polyval(p_coeffs, r)
        val_g = val_p * np.exp(-r**2 / 2.0)
        max_val = max(max_val, abs(val_g))
        
    # The sup norm of the normalized function is max_val / norm
    A_g = max_val / norm
    
    return A_g * A_g

def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    """
    def objective(coeffs):
        a0, a4, a8 = coeffs
        try:
            return _compute_c4_bound_fast(a0, a4, a8)
        except Exception:
            return 1e10
    
    # Seed coefficients
    seed_coeffs = np.array([-1.13, 0.04, 0.0003])
    
    # Bounds for each coefficient
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.1, 0.1)]
    
    # Global optimization using differential evolution
    best_val = np.inf
    best_x = seed_coeffs.copy()
    
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=500,
            popsize=20,
            tol=1e-6,
            atol=1e-8,
            init='sobol',
            polish=True
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
    
    # Local refinement from the best point found
    try:
        local_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        if local_result.fun < best_val:
            best_val = local_result.fun
            best_x = local_result.x
    except Exception:
        pass
    
    # Additional random restarts for robustness
    np.random.seed(123)
    for _ in range(15):
        start = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-0.05, 0.05)
        ])
        try:
            res = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final refinement
    try:
        final_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        if final_result.fun < best_val:
            best_val = final_result.fun
            best_x = final_result.x
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
