import time
from sympy import symbols, Rational, L, diff, Matrix, real_roots, div

def _evaluate_z(z_list):
    """Evaluate B for a given list of 12 double-root locations."""
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build Laguerre polynomials L_{2k}^{alpha}(x) for k=0..25
    laguerres = [L(2*k, alpha, x) for k in range(26)]
    
    # Unknowns c_0..c_25
    c = symbols('c0:26')
    
    # g(x) = sum c_k * L_{2k}(x)
    g = sum(c[i] * laguerres[i] for i in range(26))
    gp = diff(g, x)
    
    # Build 26x26 system
    eqs = []
    eqs.append(g.subs(x, 0))          # g(0) = 0
    eqs.append(gp.subs(x, 0) - 1)     # g'(0) = 1
    for z in z_list:
        z_r = Rational(z).limit_denominator(10**6)
        eqs.append(g.subs(x, z_r))    # g(z_i) = 0
        eqs.append(gp.subs(x, z_r))   # g'(z_i) = 0
    
    A = Matrix([[eq.diff(c[j]) for j in range(26)] for eq in eqs])
    b = Matrix([-eq for eq in eqs])
    
    try:
        sol = A.LUsolve(b)
    except Exception:
        return float('inf')
    
    # Substitute solution
    g_sub = g
    for i in range(26):
        g_sub = g_sub.subs(c[i], sol[i])
    
    # q(x) = g(x) / (x * prod(x - z_i)^2)
    denom = x
    for z in z_list:
        z_r = Rational(z).limit_denominator(10**6)
        denom = denom * (x - z_r)**2
    
    try:
        num, rem = div(g_sub, denom, x)
        if rem != 0:
            return float('inf')
        q_poly = num
    except Exception:
        return float('inf')
    
    # Find real roots
    try:
        roots = real_roots(q_poly)
    except Exception:
        return float('inf')
    
    B = 0
    for r in roots:
        if not r.is_real:
            continue
        r_val = float(r)
        if r_val <= 0:
            continue
        # Check if sign changes (simple root)
        try:
            # Use a small epsilon for sign check
            eps = Rational(1, 10**15)
            left = q_poly.subs(x, r - eps)
            right = q_poly.subs(x, r + eps)
            if left * right < 0:
                B = max(B, r_val)
        except Exception:
            pass
    
    return B

def construct():
    start_time = time.time()
    deadline = start_time + 600
    
    candidates = [
        [float(i) for i in range(1, 13)],
        [0.1 * i for i in range(1, 13)],
        [1.0 / i for i in range(1, 13)],
        [0.5 * i for i in range(1, 13)],
        [2.0 * i for i in range(1, 13)],
    ]
    
    best_B = float('inf')
    best_z = candidates[0]
    
    for z_list in candidates:
        if time.time() - start_time > deadline - 60:
            break
        try:
            B = _evaluate_z(z_list)
            if B < best_B:
                best_B = B
                best_z = z_list
        except Exception:
            pass
    
    return best_z
