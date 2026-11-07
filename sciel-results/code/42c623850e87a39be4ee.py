"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time
import sys

def _H(n, x):
    """Physicists' Hermite polynomial H_n(x) evaluated at x (array-compatible)."""
    if n == 0:
        return np.ones_like(x, dtype=float) if np.isscalar(x) is False else 1.0
    elif n == 1:
        return 2.0 * x
    else:
        # Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        H0 = np.ones_like(x, dtype=float) if np.isscalar(x) is False else 1.0
        H1 = 2.0 * x
        if n == 0:
            return H0
        if n == 1:
            return H1
        for k in range(2, n + 1):
            H2 = 2.0 * x * H1 - 2.0 * (k - 1) * H0
            H0 = H1
            H1 = H2
        return H1

def _build_g(c0, c1, c2, x):
    """Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12 with g(0)=0."""
    H0 = _H(0, x)
    H4 = _H(4, x)
    H8 = _H(8, x)
    H12 = _H(12, x)
    # H_n(0) for even n: H_0(0)=1, H_4(0)=3, H_8(0)=105, H_12(0)=10395
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12
    return g

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    # Build q(x) = g(x)/x^2. Since g is even and g(0)=0, g has factor x^2.
    # q(x) is a polynomial of degree 10 (since g has degree 12, divided by x^2).
    # Let's work with t = x^2, so q(x) = P(t) where P is degree 5.
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3 = t^2 - 6t + 3
    # H_8(x) = t^4 - 24t^3 + 120t^2 - 120t + 105
    # H_12(x) = t^6 - 36t^5 + 540t^4 - 3600t^3 + 6300t^2 - 2520t + 10395
    
    # g(x) = c0*1 + c1*(t^2-6t+3) + c2*(t^4-24t^3+120t^2-120t+105) + c3*(t^6-36t^5+540t^4-3600t^3+6300t^2-2520t+10395)
    # g(0) = c0 + 3*c1 + 105*c2 + 10395*c3 = 0
    
    # g(x) must be divisible by x^2 = t. So the constant term of g in terms of t must be 0.
    # Constant term: c0 + 3*c1 + 105*c2 + 10395*c3 = 0 (already enforced)
    
    # q(x) = g(x)/x^2 = g(t)/t where g(t) is the polynomial in t.
    # So q(x) = P(t) where P(t) = g(t)/t.
    
    # Let me compute the coefficients of g(t):
    # g(t) = c3*t^6 + (c2 - 36*c3)*t^5 + (120*c2 + 540*c3)*t^4 + (-24*c2 - 3600*c3)*t^3 + (c1 + 120*c2 + 6300*c3)*t^2 + (-6*c1 - 120*c2 - 2520*c3)*t + (c0 + 3*c1 + 105*c2 + 10395*c3)
    
    # The constant term is 0 by construction.
    # P(t) = g(t)/t = c3*t^5 + (c2 - 36*c3)*t^4 + (120*c2 + 540*c3)*t^3 + (-24*c2 - 3600*c3)*t^2 + (c1 + 120*c2 + 6300*c3)*t + (-6*c1 - 120*c2 - 2520*c3)
    
    # We need the largest real root r of q(x) at which q changes sign.
    # q(x) = P(x^2), so roots of q are at x = ±sqrt(t_i) where t_i are positive real roots of P.
    # q changes sign at x=r if P changes sign at t=r^2.
    
    # So we need: find positive real roots t_i of P(t) where P changes sign, then r = sqrt(max(t_i)).
    
    c3 = -(c0 + 3.0*c1 + 105.0*c2) / 10395.0
    
    # Coefficients of P(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    a5 = c3
    a4 = c2 - 36.0 * c3
    a3 = 120.0 * c2 + 540.0 * c3
    a2 = -24.0 * c2 - 3600.0 * c3
    a1 = c1 + 120.0 * c2 + 6300.0 * c3
    a0 = -6.0 * c1 - 120.0 * c2 - 2520.0 * c3
    
    coeffs = [a5, a4, a3, a2, a1, a0]
    
    # Find all real roots of P(t)
    try:
        roots = np.roots(coeffs)
    except:
        return float('inf')
    
    # Filter real roots (imaginary part close to 0)
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10:
            real_roots.append(rt.real)
    
    # We need positive real roots where P changes sign
    # P changes sign at a root if the root has odd multiplicity
    # For simplicity, let's check sign change numerically
    
    valid_t = []
    for t in real_roots:
        if t <= 0:
            continue
        # Check sign change: evaluate P at t-eps and t+eps
        eps = 1e-8
        P_minus = a5*(t-eps)**5 + a4*(t-eps)**4 + a3*(t-eps)**3 + a2*(t-eps)**2 + a1*(t-eps) + a0
        P_plus = a5*(t+eps)**5 + a4*(t+eps)**4 + a3*(t+eps)**3 + a2*(t+eps)**2 + a1*(t+eps) + a0
        if P_minus * P_plus < 0:
            valid_t.append(t)
    
    if not valid_t:
        return float('inf')
    
    max_t = max(valid_t)
    r = np.sqrt(max_t)
    value = r**2 / (2.0 * np.pi)
    return value

def construct():
    """Search for coefficients that minimize the value."""
    deadline = time.time() + 180.0  # 180 second limit
    
    # Start with the seed
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    # Use differential evolution for global search
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    def objective(c):
        return _compute_value(c[0], c[1], c[2])
    
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15, tol=1e-8, polish=True)
        if result.success and result.fun < best_val:
            best_c = result.x
            best_val = result.fun
    except:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_val:
                best_c = result.x
                best_val = result.fun
        except:
            pass
    
    # More local refinements with different starting points
    if time.time() < deadline - 15:
        # Try some heuristic starting points
        for start in [
            np.array([-1.0, 0.1, 0.001]),
            np.array([-2.0, 0.2, 0.002]),
            np.array([-0.5, 0.05, 0.0005]),
            np.array([-1.5, 0.15, 0.0015]),
        ]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
                if result.fun < best_val:
                    best_c = result.x
                    best_val = result.fun
            except:
                pass
    
    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
