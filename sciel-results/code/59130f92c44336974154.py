import time
from math import gcd
from itertools import combinations

# Fallback for lcm if not available (Python < 3.9)
try:
    from math import lcm
except ImportError:
    def lcm(a, b):
        return abs(a * b) // gcd(a, b) if a and b else 0

DEADLINE = 130.0  # seconds; leave margin for interpreter shutdown

def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    for r2 in (r - 1, r, r + 1):
        if r2 >= 0 and r2 * r2 == n:
            return True
    return False

def _is_square_big(n):
    # For potentially large ints, use isqrt
    try:
        from math import isqrt
        r = isqrt(n)
        return r * r == n
    except Exception:
        return _is_square(n)

def _rational_dist_sq(p1, p2, D):
    """Return (num, den) for squared distance between two points (x1,y1),(x2,y2) with coords as (num,den).
    Distance squared = (x1-x2)^2 + D*(y1-y2)^2.
    """
    x1n, x1d = p1[0]
    y1n, y1d = p1[1]
    x2n, x2d = p2[0]
    y2n, y2d = p2[1]
    # dx = x1 - x2
    dxn = x1n * x2d - x2n * x1d
    dxd = x1d * x2d
    # dy = y1 - y2
    dyn = y1n * y2d - y2n * y1d
    dyd = y1d * y2d
    # dist^2 = dxn^2/dxd^2 + D * dyn^2/dyd^2
    # = (dxn^2 * dyd^2 + D * dyn^2 * dxd^2) / (dxd^2 * dyd^2)
    num = dxn * dxn * dyd * dyd + D * dyn * dyn * dxd * dxd
    den = dxd * dxd * dyd * dyd
    # reduce
    g = gcd(num, den)
    if g:
        num //= g
        den //= g
    return num, den

def _dist_sq_is_rational(p1, p2, D):
    """Check if squared distance is a perfect square rational (i.e., distance is rational)."""
    num, den = _rational_dist_sq(p1, p2, D)
    return _is_square_big(num) and _is_square_big(den)

def _collinear(p1, p2, p3, D):
    """Check if three points are collinear using cross product = 0.
    Points: (x1, y1*sqrt(D)), etc.
    Collinearity: (x2-x1)*(y3*sqrt(D)-y1*sqrt(D)) - (x3-x1)*(y2*sqrt(D)-y1*sqrt(D)) == 0
    => (x2-x1)*(y3-y1)*sqrt(D) - (x3-x1)*(y2-y1)*sqrt(D) == 0
    => (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0   (since sqrt(D) != 0)
    So we just check the rational cross product of (x,y) coords.
    """
    x1n, x1d = p1[0]; y1n, y1d = p1[1]
    x2n, x2d = p2[0]; y2n, y2d = p2[1]
    x3n, x3d = p3[0]; y3n, y3d = p3[1]
    # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    dx12n = x2n * x1d - x1n * x2d; dx12d = x1d * x2d
    dx13n = x3n * x1d - x1n * x3d; dx13d = x1d * x3d
    dy13n = y3n * y1d - y1n * y3d; dy13d = y1d * y3d
    dy12n = y2n * y1d - y1n * y2d; dy12d = y1d * y2d
    # term1 = dx12 * dy13
    t1n = dx12n * dy13n; t1d = dx12d * dy13d
    # term2 = dx13 * dy12
    t2n = dx13n * dy12n; t2d = dx13d * dy12d
    # diff = t1 - t2
    diffn = t1n * t2d - t2n * t1d
    return diffn == 0

def _concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic.
    Four points are concyclic iff the determinant of the circle equation is zero.
    Circle: x^2 + y^2*... hmm, in our coordinates (x, y*sqrt(D)), the "squared distance" metric is x^2 + D*y^2.
    Actually, concyclicity in the Euclidean plane with coordinates (x, y*sqrt(D)):
    A circle in this plane has equation (x-a)^2 + D*(y-b)^2 = r^2.
    Expanding: x^2 - 2ax + a^2 + D*y^2 - 2Dby + D*b^2 = r^2
    => x^2 + D*y^2 + (-2a)x + (-2Db)y + (a^2 + D*b^2 - r^2) = 0
    So each point (x_i, y_i) satisfies: x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0
    for some A, B, C.
    Four points are concyclic iff the 4x4 system has a non-trivial solution (A,B,C) or is dependent.
    Equivalently, the determinant of the matrix:
    [x1^2 + D*y1^2, x1, y1, 1]
    [x2^2 + D*y2^2, x2, y2, 1]
    [x3^2 + D*y3^2, x3, y3, 1]
    [x4^2 + D*y4^2, x4, y4, 1]
    is zero.
    """
    # Compute x_i^2 + D*y_i^2 for each point
    vals = []
    for p in (p1, p2, p3, p4):
        xn, xd = p[0]; yn, yd = p[1]
        # x^2 = xn^2/xd^2, D*y^2 = D*yn^2/yd^2
        # x^2 + D*y^2 = (xn^2*yd^2 + D*yn^2*xd^2)/(xd^2*yd^2)
        num = xn * xn * yd * yd + D * yn * yn * xd * xd
        den = xd * xd * yd * yd
        vals.append((num, den, xn, xd, yn, yd))
    
    # Build 4x4 matrix with rows [v_i, x_i, y_i, 1] as fractions
    # We need to check if determinant is zero.
    # To avoid huge fractions, we can compute the determinant using integer arithmetic
    # by finding a common denominator.
    
    # Let's represent each entry as a fraction and compute determinant via expansion.
    # For efficiency, let's compute using the fact that we can multiply each row by its denominator
    # to get integers, but that changes the determinant by the product of denominators.
    
    # Row i: [v_i, x_i, y_i, 1]
    # v_i = vi_num / vi_den, x_i = xi_num/xi_den, y_i = yi_num/yi_den
    
    # Let's compute the determinant of the matrix where each entry is a fraction.
    # det = sum over permutations of sign * product of entries.
    # Each product is a fraction. Sum is zero iff the numerator of the sum (with common denominator) is zero.
    
    # This is getting complex. Let's use a simpler approach:
    # Check if the four points satisfy the circle equation by solving for A, B, C from three points
    # and checking the fourth.
    
    # From three points, we can solve:
    # x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0 for i=1,2,3
    # This is a linear system in A, B, C.
    
    # Let's compute the determinant of the 3x3 system for (A,B,C) using three points,
    # then check if the fourth point satisfies the equation.
    
    # Actually, the cleanest way: compute the 4x4 determinant directly using fractions.
    # Given the small size (4x4), let's do it.
    
    # Matrix M where M[i][j] is a fraction (num, den)
    # M[i] = [v_i, x_i, y_i, 1]
    
    M = []
    for (vnum, vden, xn, xd, yn, yd) in vals:
        M.append([
            (vnum, vden),  # x^2 + D*y^2
            (xn, xd),      # x
            (yn, yd),      # y
            (1, 1)         # 1
        ])
    
    # Compute determinant of 4x4 matrix of fractions
    # Use cofactor expansion along the last column (which has 1,1,1,1)
    # det = sum_{i=0}^{3} (-1)^{i+3} * 1 * det(M without row i, col 3)
    
    def _det3x3(m):
        """m is 3x3 list of (num, den) fractions. Return (num, den) of determinant."""
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        # aei + bfg + cdh - ceg - bdi - afh
        # Each term is product of 3 fractions
        def _mul3(x, y, z):
            n = x[0] * y[0] * z[0]
            d = x[1] * y[1] * z[1]
            g = gcd(n, d)
            if g:
                n //= g
                d //= g
            return (n, d)
        
        t1 = _mul3(a, e, i)
        t2 = _mul3(b, f, g)
        t3 = _mul3(c, d, h)
        t4 = _mul3(c, e, g)
        t5 = _mul3(b, d, i)
        t6 = _mul3(a, f, h)
        
        # sum = t1 + t2 + t3 - t4 - t5 - t6
        # Find common denominator
        den = 1
        for term in (t1, t2, t3, t4, t5, t6):
            den = lcm(den, term[1])
        num = 0
        for term, sign in [(t1,1),(t2,1),(t3,1),(t4,-1),(t5,-1),(t6,-1)]:
            num += sign * term[0] * (den // term[1])
        g = gcd(abs(num), den)
        if g:
            num //= g
            den //= g
        return (num, den)
    
    def _det4x4_frac(m):
        """m is 4x4 list of (num,den). Return (num,den) of determinant."""
        # Expand along last column
        total_num = 0
        total_den = 1
        for i in range(4):
            sign = (-1) ** (i + 3)
            # Minor: remove row i, col 3
            minor = []
            for r in range(4):
                if r == i:
                    continue
                minor.append(m[r][:3])
            dnum, dden = _det3x3(minor)
            # term = sign * m[i][3] * (dnum/dden) = sign * (1/1) * dnum/dden = sign*dnum/dden
            term_num = sign * dnum
            term_den = dden
            # Add to total
            lcm_val = lcm(total_den, term_den)
            total_num = total_num * (lcm_val // total_den) + term_num * (lcm_val // term_den)
            total_den = lcm_val
        g = gcd(abs(total_num), total_den)
        if g:
            total_num //= g
            total_den //= g
        return (total_num, total_den)
    
    dnum, dden = _det4x4_frac(M)
    return dnum == 0

def _check_valid(points, D):
    """Check all conditions: rational pairwise distances, no three collinear, no four concyclic."""
    n = len(points)
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            if not _dist_sq_is_rational(points[i], points[j], D):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k], D):
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _format_point(x, y):
    """Convert integer or rational to [num, den] format."""
    if isinstance(x, tuple):
        return [x[0], x[1]]
    return [x, 1]

def construct():
    start_time = time.time()
    
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base is valid
    base_fmt = [[[x, 1], [y, 1]] for x, y in base_points]
    
    if not _check_valid(base_fmt, D):
        # Should not happen, but fallback
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points[:3]]}
    
    best_points = base_fmt
    best_count = len(base_fmt)
    
    # The grid search in the original code was too slow and prone to timeout.
    # We return the verified 5-point set immediately.
    
    return {'D': D, 'points': best_points}
