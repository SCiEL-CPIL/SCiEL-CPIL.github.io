import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _compute_score(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots
    from sympy import Matrix, binomial, factorial
    
    # Convert z to rationals
    z_rationals = [Rational(float(z)) for z in z_list]
    z_rationals.sort()
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    def laguerre_generalized(n, a, x):
        """Generalized Laguerre polynomial L_n^(a)(x)"""
        result = 0
        for k in range(n + 1):
            result += (-1)**k * binomial(n + a, n - k) * x**k / factorial(k)
        return result
    
    # Degrees: 0, 2, 4, ..., 50 (26 terms)
    degrees = [2*k for k in range(26)]
    
    # g(x) = sum_k c_k * L_{2k}^{(-1/2)}(x)
    c = [symbols(f'c_{i}') for i in range(26)]
    
    g = 0
    for k, deg in enumerate(degrees):
        Lpoly = laguerre_generalized(deg, alpha, x)
        g += c[k] * Lpoly
    
    g = g.expand()
    g_prime = diff(g, x)
    
    # Constraints:
    # g(0) = 0
    # g'(0) = 1
    # g(z_i) = 0 for each i
    # g'(z_i) = 0 for each i
    
    equations = []
    equations.append(g.subs(x, 0))  # g(0) = 0
    equations.append(g_prime.subs(x, 0) - 1)  # g'(0) = 1
    
    for zi in z_rationals:
        equations.append(g.subs(x, zi))  # g(z_i) = 0
        equations.append(g_prime.subs(x, zi))  # g'(z_i) = 0
    
    # 2 + 2*12 = 26 equations for 26 unknowns
    if len(equations) != 26:
        return None
    
    # Build linear system
    coeff_matrix = []
    rhs = []
    for eq in equations:
        eq = eq.expand()
        row = []
        rest = eq
        for i in range(26):
            coeff = rest.coeff(c[i])
            rest = rest - coeff * c[i]
            row.append(coeff)
        if rest != 0:
            return None
        coeff_matrix.append(row)
        rhs.append(0)
    
    # Solve
    A = Matrix(coeff_matrix)
    b = Matrix(rhs)
    
    try:
        sol = A.LUsolve(b)
    except Exception:
        return None
    
    # Check for singular
    if any(v is None or v.has(symbols('c')) for v in sol):
        return None
    
    # Build g with solved coefficients
    g_solved = 0
    for k, deg in enumerate(degrees):
        Lpoly = laguerre_generalized(deg, alpha, x)
        g_solved += sol[k] * Lpoly
    g_solved = g_solved.expand()
    
    # q(x) = g(x) / (x * prod_i (x - z_i)^2)
    denom = x
    for zi in z_rationals:
        denom *= (x - zi)**2
    denom = denom.expand()
    
    # Polynomial division
    try:
        q_poly, rem = Poly(g_solved, x).div(Poly(denom, x))
    except Exception:
        return None
    
    if rem != 0:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    real_roots_list = real_roots(q_expr, x)
    
    B = 0
    for r in real_roots_list:
        r_val = float(r)
        if r_val <= 0:
            continue
        # Check sign change at r ± 1e-198
        r_minus = r - Rational(1, 10)**198
        r_plus = r + Rational(1, 10)**198
        q_minus = q_expr.subs(x, r_minus)
        q_plus = q_expr.subs(x, r_plus)
        if q_minus * q_plus < 0:
            if r_val > B:
                B = r_val
    
    return B / (2 * math.pi)


def _get_laguerre_zeros(n, alpha_val):
    """Get the n positive zeros of L_n^(alpha)(x) using scipy."""
    from scipy.special import roots_genlaguerre
    # roots_genlaguerre returns (nodes, weights) for L_n^(alpha)
    nodes, weights = roots_genlaguerre(n, alpha_val)
    return [float(x) for x in nodes]


def construct():
    """Return 12 double-root locations z_1..z_12."""
    start_time = time.time()
    deadline = start_time + 570.0  # 60s safety margin
    
    def time_remaining():
        return deadline - time.time()
    
    # Structural change: Use zeros of L_12^(-1/2)(x) as primary starting point
    # These are the natural nodes for the Laguerre quadrature with weight x^(-1/2) * e^(-x)
    try:
        z_laguerre = _get_laguerre_zeros(12, -0.5)
    except Exception:
        z_laguerre = [i**2 * math.pi**2 for i in range(1, 13)]
    
    # Also prepare the old base case as fallback
    z_base = [i**2 * math.pi**2 for i in range(1, 13)]
    
    best_score = float('inf')
    best_z = z_laguerre[:]
    
    # Try the Laguerre zeros first (structural change)
    try:
        s = _compute_score(z_laguerre)
        if s is not None and s < best_score:
            best_score = s
            best_z = z_laguerre[:]
    except Exception:
        pass
    
    # Try the old base case
    try:
        s = _compute_score(z_base)
        if s is not None and s < best_score:
            best_score = s
            best_z = z_base[:]
    except Exception:
        pass
    
    if time_remaining() < 60:
        return best_z
    
    # Try scaled versions of Laguerre zeros
    for scale in [0.8, 0.9, 1.1, 1.2, 0.7, 1.3]:
        if time_remaining() < 120:
            break
        z_try = [scale * z for z in z_laguerre]
        if any(z <= 0 for z in z_try):
            continue
        if len(set(z_try)) != 12:
            continue
        try:
            s = _compute_score(z_try)
            if s is not None and s < best_score:
                best_score = s
                best_z = z_try[:]
        except Exception:
            pass
    
    if time_remaining() < 60:
        return best_z
    
    # Try shifted versions: z_i = z_laguerre[i] + offset
    for offset in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
        if time_remaining() < 120:
            break
        z_try = [z + offset for z in z_laguerre]
        if any(z <= 0 for z in z_try):
            continue
        if len(set(z_try)) != 12:
            continue
        try:
            s = _compute_score(z_try)
            if s is not None and s < best_score:
                best_score = s
                best_z = z_try[:]
        except Exception:
            pass
    
    if time_remaining() < 60:
        return best_z
    
    # Try mixed: some Laguerre zeros, some (i*pi)^2
    for frac in [0.5, 0.6, 0.4]:
        if time_remaining() < 120:
            break
        n_lag = int(12 * frac)
        z_try = z_laguerre[:n_lag] + [z_base[i] for i in range(n_lag, 12)]
        if any(z <= 0 for z in z_try):
            continue
        if len(set(z_try)) != 12:
            continue
        try:
            s = _compute_score(z_try)
            if s is not None and s < best_score:
                best_score = s
                best_z = z_try[:]
        except Exception:
            pass
    
    if time_remaining() < 60:
        return best_z
    
    # Fine-tune around best
    for delta in [0.05, -0.05, 0.1, -0.1]:
        if time_remaining() < 120:
            break
        z_try = [z * (1 + delta) for z in best_z]
        if any(z <= 0 for z in z_try):
            continue
        if len(set(z_try)) != 12:
            continue
        try:
            s = _compute_score(z_try)
            if s is not None and s < best_score:
                best_score = s
                best_z = z_try[:]
        except Exception:
            pass
    
    # Validate result
    result = best_z
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [i**2 * math.pi**2 for i in range(1, 13)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [i**2 * math.pi**2 for i in range(1, 13)]
                break
    
    return result
# EVOLVE-BLOCK-END
