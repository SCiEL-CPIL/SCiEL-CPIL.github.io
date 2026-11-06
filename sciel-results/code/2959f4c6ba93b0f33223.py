import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Find a set of points (x, y*sqrt(D)) with rational x,y, squarefree D,
    such that all pairwise distances are rational, no three collinear,
    no four concyclic. Maximize number of points.
    """
    t0 = time.time()
    deadline = 150.0  # seconds

    # We'll search over small squarefree D values and try to build
    # large sets of points with rational pairwise distances.
    #
    # Strategy: For D=1, we need integer (or rational) points with
    # integer pairwise distances. This is the classic "rational distance
    # set" problem. Known results:
    # - 5 points exist (e.g., the one in the seed)
    # - 6 points exist (known constructions)
    # - 7 points: open / very hard
    #
    # For D > 1, the problem becomes: find rational (x_i, y_i) such that
    # (x_i - x_j)^2 + D*(y_i - y_j)^2 is a perfect square in Q for all i,j.
    #
    # A powerful approach: use the parametrization of conics.
    # If we fix one point at origin, then for each other point (x,y),
    # we need x^2 + D*y^2 = r^2 for some rational r.
    # This is a conic (hyperbola or ellipse depending on sign), which has
    # rational parametrizations.
    #
    # Better approach: Use known constructions and extend them.
    #
    # Let's try a systematic search for D=1 first (integer points with
    # integer distances), then try small D values.
    
    # Helper: check if a set of points is valid
    def check_valid(D, points):
        """
        points: list of (Fraction x, Fraction y) representing (x, y*sqrt(D))
        Returns True if all pairwise distances are rational, no three collinear,
        no four concyclic.
        """
        n = len(points)
        # Check pairwise distances are rational
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                # squared distance = dx^2 + D * dy^2
                d2 = dx*dx + D*dy*dy
                # Check if d2 is a perfect square in Q
                # d2 = p/q in lowest terms, need p and q to be perfect squares
                if d2 < 0:
                    return False
                p = d2.numerator
                q = d2.denominator
                # Check p is perfect square
                sp = math.isqrt(p)
                if sp*sp != p:
                    return False
                sq = math.isqrt(q)
                if sq*sq != q:
                    return False
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0
            cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
            if cross == 0:
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            x4, y4 = points[l]
            # Four points are concyclic iff the determinant of the circle equation is zero
            # Circle: x^2 + D*y^2 + Ax + By + C = 0
            # For each point: x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0
            # This is a system of 4 equations in A, B, C. Consistent iff det = 0.
            # Build matrix:
            # [x1^2 + D*y1^2, x1, y1, 1]
            # [x2^2 + D*y2^2, x2, y2, 1]
            # [x3^2 + D*y3^2, x3, y3, 1]
            # [x4^2 + D*y4^2, x4, y4, 1]
            # det should be non-zero for non-concyclic
            M = [
                [x1*x1 + D*y1*y1, x1, y1, Fraction(1)],
                [x2*x2 + D*y2*y2, x2, y2, Fraction(1)],
                [x3*x3 + D*y3*y3, x3, y3, Fraction(1)],
                [x4*x4 + D*y4*y4, x4, y4, Fraction(1)],
            ]
            if _det4(M) == 0:
                return False
        return True

    def _det4(M):
        """Compute determinant of 4x4 matrix of Fractions using cofactor expansion."""
        # Use Bareiss algorithm for exact arithmetic
        # Bareiss's fraction-free Gaussian elimination
        n = 4
        A = [row[:] for row in M]
        sign = Fraction(1)
        prev = Fraction(1)
        for k in range(n-1):
            # Find pivot
            if A[k][k] == 0:
                swap = None
                for i in range(k+1, n):
                    if A[i][k] != 0:
                        swap = i
                        break
                if swap is None:
                    return Fraction(0)
                A[k], A[swap] = A[swap], A[k]
                sign = -sign
            pivot = A[k][k]
            for i in range(k+1, n):
                for j in range(k+1, n):
                    A[i][j] = (A[i][j]*pivot - A[i][k]*A[k][j]) // prev
            prev = pivot
        return sign * A[n-1][n-1]

    # Strategy: Build sets incrementally.
    # Start with known good configurations and try to extend.
    
    # Known 5-point set for D=1:
    base5 = [(Fraction(0), Fraction(0)),
             (Fraction(-56), Fraction(33)),
             (Fraction(-16), Fraction(63)),
             (Fraction(0), Fraction(33)),
             (Fraction(16), Fraction(63))]
    
    best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base5]}
    best_count = 5
    
    # Try to extend to 6 points for D=1
    # Search for a 6th point (x, y) with integer coordinates such that
    # distance to all 5 existing points is integer.
    
    def try_extend(D, existing, max_coord=200):
        """Try to add one more point to existing set."""
        n = len(existing)
        for x in range(-max_coord, max_coord+1):
            for y in range(-max_coord, max_coord+1):
                new_pt = (Fraction(x), Fraction(y))
                # Check distance to all existing points
                ok = True
                for ex, ey in existing:
                    dx = new_pt[0] - ex
                    dy = new_pt[1] - ey
                    d2 = dx*dx + D*dy*dy
                    p = d2.numerator
                    q = d2.denominator
                    sp = math.isqrt(p)
                    sq = math.isqrt(q)
                    if sp*sp != p or sq*sq != q:
                        ok = False
                        break
                if not ok:
                    continue
                # Check collinearity with all pairs
                collinear = False
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1 = existing[i]
                        x2, y2 = existing[j]
                        cross = (x2-x1)*(new_pt[1]-y1) - (y2-y1)*(new_pt[0]-x1)
                        if cross == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                # Check concyclicity with all triples
                concyclic = False
                for i, j, k in combinations(range(n), 3):
                    x1, y1 = existing[i]
                    x2, y2 = existing[j]
                    x3, y3 = existing[k]
                    x4, y4 = new_pt
                    M = [
                        [x1*x1 + D*y1*y1, x1, y1, Fraction(1)],
                        [x2*x2 + D*y2*y2, x2, y2, Fraction(1)],
                        [x3*x3 + D*y3*y3, x3, y3, Fraction(1)],
                        [x4*x4 + D*y4*y4, x4, y4, Fraction(1)],
                    ]
                    if _det4(M) == 0:
                        concyclic = True
                        break
                if concyclic:
                    continue
                return new_pt
        return None

    # Try extending the 5-point set to 6
    if time.time() < deadline - 30:
        ext = try_extend(1, base5, max_coord=150)
        if ext is not None:
            new_set = base5 + [ext]
            if check_valid(1, new_set):
                best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in new_set]}
                best_count = 6
                print(f"Found 6-point set: {new_set}")
    
    # Try other D values with systematic search
    # For small D, try to build sets from scratch using parametrization
    
    # For D=2: x^2 + 2y^2 = r^2
    # Parametrization: let t = x/y (rational), then x = t*y, so y^2(t^2+2) = r^2
    # This requires t^2+2 to be a square times a square... not straightforward.
    # Alternative: use Pythagorean triples generalized.
    
    # Let's try a different approach: for each D, generate candidate points
    # on the "rational distance" locus and try to assemble a set.
    
    # For D=1, also try other known 5-point configurations and extend them.
    
    # Try a grid search for 6-point sets directly for D=1
    if time.time() < deadline - 20:
        # Generate all integer points in a small region with integer mutual distances
        # This is expensive, so use a smarter approach:
        # Start with pairs that have integer distance, build up.
        
        # Actually, let's try known 6-point constructions.
        # One known 6-point rational distance set:
        # (0,0), (0,4), (3,0), (3,4), (0,5), (5,0)? Let's check...
        # Distances: (0,0)-(0,4)=4, (0,0)-(3,0)=3, (0,0)-(3,4)=5, 
        # (0,0)-(0,5)=5, (0,0)-(5,0)=5
        # (0,4)-(3,0)=5, (0,4)-(3,4)=3, (0,4)-(0,5)=1, (0,4)-(5,0)=sqrt(41) no
        
        # Let me try: (0,0), (3,4), (5,12), (12,5), (15,8), (8,15)?
        # These are all at distance from origin: 5, 13, 13, 17, 17... no
        
        # Let's just do a targeted search with the extend function on different bases.
        
        # Try another 5-point base
        base5b = [(Fraction(0), Fraction(0)),
                  (Fraction(3), Fraction(4)),
                  (Fraction(5), Fraction(12)),
                  (Fraction(0), Fraction(13)),
                  (Fraction(12), Fraction(5))]
        # Check if this is valid
        if check_valid(1, base5b):
            if time.time() < deadline - 20:
                ext2 = try_extend(1, base5b, max_coord=100)
                if ext2 is not None:
                    new_set2 = base5b + [ext2]
                    if check_valid(1, new_set2):
                        best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in new_set2]}
                        best_count = 6
                        print(f"Found 6-point set (base b): {new_set2}")
    
    # Try D=2, 3, 5, 6, 7, 10, 11, 13, 14, 15 with small searches
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47]:
        if time.time() > deadline - 15:
            break
        # Try to build a 5-point set for this D
        # Start with origin and add points one by one
        pts = [(Fraction(0), Fraction(0))]
        found = False
        for target_size in range(2, 6):
            if time.time() > deadline - 15:
                break
            added = False
            # Try to find a point that works with all existing
            for x in range(-30, 31):
                for y in range(-30, 31):
                    if x == 0 and y == 0:
                        continue
                    new_pt = (Fraction(x), Fraction(y))
                    ok = True
                    for ex, ey in pts:
                        dx = new_pt[0] - ex
                        dy = new_pt[1] - ey
                        d2 = dx*dx + D*dy*dy
                        if d2 < 0:
                            ok = False
                            break
                        p = d2.numerator
                        q = d2.denominator
                        sp = math.isqrt(p)
                        sq = math.isqrt(q)
                        if sp*sp != p or sq*sq != q:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check collinearity
                    collinear = False
                    for i in range(len(pts)):
                        for j in range(i+1, len(pts)):
                            x1, y1 = pts[i]
                            x2, y2 = pts[j]
                            cross = (x2-x1)*(new_pt[1]-y1) - (y2-y1)*(new_pt[0]-x1)
                            if cross == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if collinear:
                        continue
                    # Check concyclicity
                    concyclic = False
                    if len(pts) >= 3:
                        for i, j, k in combinations(range(len(pts)), 3):
                            x1, y1 = pts[i]
                            x2, y2 = pts[j]
                            x3, y3 = pts[k]
                            x4, y4 = new_pt
                            M = [
                                [x1*x1 + D*y1*y1, x1, y1, Fraction(1)],
                                [x2*x2 + D*y2*y2, x2, y2, Fraction(1)],
                                [x3*x3 + D*y3*y3, x3, y3, Fraction(1)],
                                [x4*x4 + D*y4*y4, x4, y4, Fraction(1)],
                            ]
                            if _det4(M) == 0:
                                concyclic = True
                                break
                    if concyclic:
                        continue
                    pts.append(new_pt)
                    added = True
                    break
                if added:
                    break
            if not added:
                break
        
        if len(pts) > best_count:
            if check_valid(D, pts):
                best = {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]}
                best_count = len(pts)
                print(f"Found {best_count}-point set for D={D}")
    
    # Final safety: return best found
    return best

# EVOLVE-BLOCK-END
