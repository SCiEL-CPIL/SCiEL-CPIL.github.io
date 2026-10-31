"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time
import sympy as sp


def _hermite_coeffs(n):
    """Get coefficients of physicist's Hermite polynomial H_n(x) as a list [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n."""
    H = hermite(n)
    coeffs = np.array(H.coeffs()[::-1], dtype=float)
    return coeffs


def _poly_mul(p, q):
    """Multiply two polynomials given as coefficient lists (lowest degree first)."""
    result = np.zeros(len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i + j] += a * b
    return result


def _poly_integrate(p, x):
    """Integrate polynomial p (coeffs lowest first) from 0 to x, return coeffs of antiderivative."""
    n = len(p)
    result = np.zeros(n + 1)
    for i, c in enumerate(p):
        result[i + 1] = c / (i + 1)
    return result


def _poly_eval(p, x):
    """Evaluate polynomial at x using Horner's method."""
    result = 0.0
    for c in reversed(p):
        result = result * x + c
    return result


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty bound using exact polynomial arithmetic.
    
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    The Hermite functions φ_n(x) = H_n(x)*exp(-x^2/2)/sqrt(2^n*n!*sqrt(pi)) are orthonormal.
    
    For g(x) = sum_n c_n * H_n(x) * exp(-x^2/2):
    ||g||^2 = sum_n c_n^2 * 2^n * n! * sqrt(pi)
    
    <x^2> = integral x^2 |g|^2 / integral |g|^2
    
    We need: integral x^2 * H_m(x)*H_n(x) * exp(-x^2) dx
    
    Using the identity: x*H_n = H_{n+1}/2 + n*H_{n-1}
    So x^2*H_n = (H_{n+2} + (2n+1)*H_n + (n-1)*H_{n-2})/4
    
    Therefore: integral x^2 * H_m * H_n * exp(-x^2) dx
    = sum_k [(delta_{m,k+2} + (2k+1)*delta_{m,k} + (k-1)*delta_{m,k-2}) / 4] * integral H_k * H_n * exp(-x^2) dx
    = sum_k [(delta_{m,k+2} + (2k+1)*delta_{m,k} + (k-1)*delta_{m,k-2}) / 4] * 2^k * k! * sqrt(pi) * delta_{m,n}
    
    Wait, that's only when m=n. Let me be more careful.
    
    integral H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_{m,n}
    
    So integral x^2 H_m H_n exp(-x^2) dx = integral H_m * [x^2 H_n] * exp(-x^2) dx
    
    x^2 H_n = (1/4)(H_{n+2} + (2n+1)H_n + (n-1)H_{n-2})  (for n>=2; handle edge cases)
    
    So integral x^2 H_m H_n exp(-x^2) = (1/4) [2^{n+2}(n+2)!sqrt(pi) delta_{m,n+2} + (2n+1)2^n n! sqrt(pi) delta_{m,n} + (n-1)2^{n-2}(n-2)!sqrt(pi) delta_{m,n-2}]
    """
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    # Compute norm: ||g||^2 = sum_n c_n^2 * 2^n * n! * sqrt(pi)
    sqrt_pi = np.sqrt(np.pi)
    norm2 = 0.0
    for c, n in zip(coeffs, degrees):
        norm2 += c**2 * (2.0**n) * (np.math.factorial(n)) * sqrt_pi
    
    if norm2 < 1e-30:
        return 1e30
    
    # Compute integral x^2 |g|^2 = sum_{m,n} c_m c_n * integral x^2 H_m H_n exp(-x^2) dx
    # = sum_{m,n} c_m c_n * (1/4) [2^{n+2}(n+2)!sqrt(pi) delta_{m,n+2} + (2n+1)2^n n! sqrt(pi) delta_{m,n} + (n-1)2^{n-2}(n-2)!sqrt(pi) delta_{m,n-2}]
    
    x2_norm = 0.0
    for i, (ci, mi) in enumerate(zip(coeffs, degrees)):
        for j, (cj, nj) in enumerate(zip(coeffs, degrees)):
            term = 0.0
            # delta_{m,n+2}: m = n+2
            if mi == nj + 2:
                term += 2.0**(nj+2) * (np.math.factorial(nj+2)) * sqrt_pi
            # (2n+1) delta_{m,n}: m = n
            if mi == nj:
                term += (2*nj + 1) * 2.0**nj * (np.math.factorial(nj)) * sqrt_pi
            # (n-1) delta_{m,n-2}: m = n-2
            if mi == nj - 2:
                term += (nj-1) * 2.0**(nj-2) * (np.math.factorial(nj-2)) * sqrt_pi
            x2_norm += ci * cj * term / 4.0
    
    return x2_norm / norm2


def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the upper bound C4.
    Uses analytic gradient + multi-start local optimization.
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s kill
    
    # Initial best from seed
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = _compute_bound_analytic(*best_c)
    
    def objective_and_grad(params):
        c0, c4, c8 = params
        val = _compute_bound_analytic(c0, c4, c8)
        
        # Compute gradient numerically with central differences
        h = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            grad[i] = (_compute_bound_analytic(*p_plus) - _compute_bound_analytic(*p_minus)) / (2*h)
        
        return val, grad
    
    # Multi-start local optimization
    np.random.seed(42)
    bounds = [(-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0)]
    
    # Generate diverse starting points
    starts = [best_c.copy()]
    
    # Add random starts in different scales
    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for _ in range(5):
            starts.append(np.random.uniform(-scale, scale, 3))
    
    # Add structured starts based on the constraint
    for t in np.linspace(-1, 1, 9):
        starts.append(np.array([t, 0.1*t, 0.01*t]))
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Use L-BFGS-B with gradient
            result = minimize(
                lambda p: _compute_bound_analytic(*p),
                start,
                jac=lambda p: None,  # Let it use finite differences if we don't provide
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final polish from the best point
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                lambda p: _compute_bound_analytic(*p),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
