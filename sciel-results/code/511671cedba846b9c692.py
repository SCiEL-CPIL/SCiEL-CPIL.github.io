"""Refined program: direct Bessel-zero construction with time-budgeted refinement."""
import time
import math

# EVOLVE-BLOCK-START

def _laguerre_zeros(n, alpha):
    """Get the n positive zeros of the generalized Laguerre polynomial L_n^(alpha)(x)."""
    try:
        import scipy.special
        roots, _ = scipy.special.roots_genlaguerre(n, alpha)
        return [float(r) for r in roots]
    except Exception:
        # Fallback to Bessel squares if scipy fails
        return [_bessel_zero_J0(i+1)**2 for i in range(n)]

def _bessel_zero_J0(k):
    """High-accuracy k-th positive zero of J_0 using asymptotic expansion."""
    if k == 1:
        return 2.4048255576957728
    if k == 2:
        return 5.5200781102863106
    if k == 3:
        return 8.6537279129110122
    if k == 4:
        return 11.7915344390143598
    if k == 5:
        return 14.9309177084877878
    if k == 6:
        return 18.0710639679109298
    if k == 7:
        return 21.2116366299713778
    if k == 8:
        return 24.3524715308462127
    if k == 9:
        return 27.4934791320444287
    if k == 10:
        return 30.6346064684319602
    if k == 11:
        return 33.7758202136374444
    if k == 12:
        return 36.9171012943291185
    # General asymptotic for larger k
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, binomial, S)
    from sympy.polys.orthopoly import genlaguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = S(-1)/2

    # Build L_{2k}^{alpha}(x) for k=0..25 using sympy's genlaguerre
    laguerre_pols = []
    for k in range(26):
        n = 2*k
        laguerre_pols.append(genlaguerre(n, alpha)(x))

    # Build system matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: S(0))
    b = Matrix(26, 1, lambda i, j: S(0))

    # Precompute values for speed
    for k in range(26):
        L = laguerre_pols[k]
        Ld = L.diff(x)
        A[0, k] = L.subs(x, 0)
        A[1, k] = Ld.subs(x, 0)
        for i in range(12):
            zi = z_rationals[i]
            A[2 + 2*i, k] = L.subs(x, zi)
            A[3 + 2*i, k] = Ld.subs(x, zi)
    b[1, 0] = 1

    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    # Denominator: x * prod (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2

    g_poly = Poly(g, x)
    denom_poly = Poly(denom)
    try:
        q_poly, rem = g_poly.div(denom_poly)
        if rem != 0:
            return None
    except Exception:
        return None

    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None

    eps = Rational(1, 10)**198
    max_r = S(0)
    for r in real_roots_list:
        try:
            qm = q_expr.subs(x, r - eps)
            qp = q_expr.subs(x, r + eps)
            if qm * qp < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue

    B = max(S(0), max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _is_valid(z_list):
    """Check if z_list is valid: finite, positive, pairwise distinct."""
    for i in range(12):
        if not math.isfinite(z_list[i]) or z_list[i] <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-9:
                return False
    return True

def construct():
    """Return 12 double-root locations using spectral Laguerre initialization."""
    deadline = time.time() + 570.0  # 60s safety margin

    # Primary construction: Zeros of L_12^(-1/2)(x)
    # This aligns the double roots with the natural nodes of the Laguerre weight
    alpha = -0.5
    z_laguerre = _laguerre_zeros(12, alpha)
    result = list(z_laguerre)

    # Validate
    if not _is_valid(result):
        result = [float(i+1) * 5.0 for i in range(12)]

    # Time-budgeted refinement: try geometric scaling and local search
    remaining = deadline - time.time()
    if remaining > 100:
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Try several geometric scalings
                scales = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
                for s in scales:
                    if time.time() > deadline - 120:
                        break
                    candidate = [z * s for z in result]
                    if not _is_valid(candidate):
                        continue
                    score = _compute_score_exact(candidate)
                    if score is not None and score < best_score:
                        best_score = score
                        result = candidate
        except Exception:
            pass

    # If we still have time, try a finer local search around the best result
    remaining = deadline - time.time()
    if remaining > 150:
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Coordinate descent with adaptive steps
                # We have ~150s left. Each eval ~5-10s. We can do ~15-20 evals.
                # We will try to adjust each z_i individually.
                steps = [0.1, 0.5, 1.0, 2.0, 5.0]
                current_z = list(result)
                
                # Try to improve each coordinate
                for i in range(12):
                    if time.time() > deadline - 60:
                        break
                    # Try different step sizes for this coordinate
                    for step in steps:
                        for direction in [1, -1]:
                            if time.time() > deadline - 60:
                                break
                            candidate = list(current_z)
                            candidate[i] += direction * step
                            if not _is_valid(candidate):
                                continue
                            score = _compute_score_exact(candidate)
                            if score is not None and score < best_score:
                                best_score = score
                                current_z = candidate
                                result = candidate
                                break # Move to next step size or coordinate? 
                                # Actually, if we found an improvement, we should restart step search for this coord
                                # But for simplicity, just accept and move on to next step/coord
                                # A more aggressive search would loop back to small steps
                                break 
                result = current_z
        except Exception:
            pass

    return result
# EVOLVE-BLOCK-END
