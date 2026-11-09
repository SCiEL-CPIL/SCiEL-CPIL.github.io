"""Search for integral point sets in general position."""
import time
import random
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _elapsed():
    return time.time() - _T0


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _is_squarefree(n):
    if n <= 0:
        return False
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _is_rational_sqrt(d2):
    """Check if a Fraction d2 is a perfect square rational.
    Return (rn, rd) such that sqrt(d2) = rn/rd, or None."""
    if d2 < 0:
        return None
    num = d2.numerator
    den = d2.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    rn = isqrt(num)
    rd = isqrt(den)
    if rn * rn == num and rd * rd == den:
        return (rn, rd)
    return None


def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)


def _four_concyclic(p1, p2, p3, p4, D):
    """Check if 4 points (x, y*sqrt(D)) are concyclic using the determinant."""
    M = []
    for (x, y) in (p1, p2, p3, p4):
        M.append([x * x + D * y * y, x, y, Fraction(1)])
    n = 4
    A = [row[:] for row in M]
    for c in range(n):
        pivot = None
        for r in range(c, n):
            if A[r][c] != 0:
                pivot = r
                break
        if pivot is None:
            return True
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for cc in range(c, n):
                A[r][cc] -= factor * A[c][cc]
    det = Fraction(1)
    for i in range(n):
        det *= A[i][i]
    return det == 0


def _check_general_position(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True


def _build_result(points, D):
    out = []
    for (x, y) in points:
        out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': out}


def _score(points, D):
    return len(points)


def _valid_distance(x, y, px, py, D):
    """Check if distance squared between (x,y) and (px,py) is a perfect square rational."""
    dx = Fraction(x) - px
    dy = Fraction(y) - py
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    return _is_rational_sqrt(d2) is not None


def _try_add_point(current, new_pt, D):
    """Try to add new_pt to current set, checking all constraints."""
    n = len(current)
    # Check distances
    for (px, py) in current:
        if not _valid_distance(new_pt[0], new_pt[1], px, py, D):
            return False
    # Check collinearity
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(current[i], current[j], new_pt):
                return False
    # Check concyclicity
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _four_concyclic(current[i], current[j], current[k], new_pt, D):
                    return False
    return True


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1

    # Start with a simple valid 3-point set: (0,0), (1,0), (0,1) with D=1
    simple_3 = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    if _check_general_position(simple_3, 1):
        best = simple_3
        best_score = 3

    # Strategy 1: Try D=3 hexagon structure and perturb to break concyclicity
    if not _deadline_hit():
        D_hex = 3
        hexagon_6 = [
            (Fraction(1), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2)),
            (Fraction(-1, 2), Fraction(1, 2)),
            (Fraction(-1), Fraction(0)),
            (Fraction(-1, 2), Fraction(-1, 2)),
            (Fraction(1, 2), Fraction(-1, 2))
        ]
        
        # Check if hexagon has rational distances with D=3
        hex_ok = True
        for i in range(6):
            for j in range(i + 1, 6):
                if not _valid_distance(hexagon_6[i][0], hexagon_6[i][1], 
                                       hexagon_6[j][0], hexagon_6[j][1], D_hex):
                    hex_ok = False
                    break
            if not hex_ok:
                break
        
        if hex_ok:
            # The hexagon is concyclic (all on a circle), so we need to perturb
            # Try perturbing one point by small rational amounts
            base = list(hexagon_6)
            found = False
            for perturb_idx in range(6):
                if _deadline_hit() or found:
                    break
                # Try different perturbation sizes
                for scale in [1, 2, 3, 4, 5]:
                    if _deadline_hit() or found:
                        break
                    for dx_n in range(-scale, scale + 1):
                        if _deadline_hit() or found:
                            break
                        for dy_n in range(-scale, scale + 1):
                            if _deadline_hit() or found:
                                break
                            if dx_n == 0 and dy_n == 0:
                                continue
                            for den in range(1, scale + 1):
                                if _deadline_hit() or found:
                                    break
                                new_pt = (
                                    base[perturb_idx][0] + Fraction(dx_n, den),
                                    base[perturb_idx][1] + Fraction(dy_n, den)
                                )
                                candidate = list(base)
                                candidate[perturb_idx] = new_pt
                                
                                # Check all pairwise distances
                                valid = True
                                for i in range(6):
                                    for j in range(i + 1, 6):
                                        if not _valid_distance(candidate[i][0], candidate[i][1],
                                                               candidate[j][0], candidate[j][1], D_hex):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    continue
                                
                                # Check general position
                                if _check_general_position(candidate, D_hex):
                                    best_score = 6
                                    best = list(candidate)
                                    found = True
                                    break
        
        # Strategy 2: If we found 6 points, try to extend to 7+
        if best_score >= 6 and not _deadline_hit():
            current = list(best)
            # Try adding more points near the existing ones
            for _attempt in range(100):
                if _deadline_hit():
                    break
                # Pick a random point near one of the existing points
                base_pt = random.choice(current)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                den = random.randint(1, 10)
                new_pt = (
                    base_pt[0] + Fraction(dx, den),
                    base_pt[1] + Fraction(dy, den)
                )
                if new_pt in current:
                    continue
                if _try_add_point(current, new_pt, D_hex):
                    current.append(new_pt)
                    best_score = len(current)
                    best = list(current)
                    break
        
        # Strategy 3: Try D=1 with integer lattice points
        if best_score < 6 and not _deadline_hit():
            random.seed(42)
            # Generate candidate integer points
            candidate_pool = []
            for x in range(-50, 51):
                for y in range(-50, 51):
                    candidate_pool.append((Fraction(x), Fraction(y)))
            
            # Greedy construction
            for start_idx in range(0, len(candidate_pool), 100):
                if _deadline_hit():
                    break
                
                current = [candidate_pool[start_idx]]
                for _ in range(10):
                    if _deadline_hit():
                        break
                    added = False
                    # Try candidates in random order
                    indices = list(range(len(candidate_pool)))
                    random.shuffle(indices)
                    for idx in indices:
                        if _deadline_hit():
                            break
                        p = candidate_pool[idx]
                        if p in current:
                            continue
                        if _try_add_point(current, p, 1):
                            current.append(p)
                            added = True
                            if len(current) > best_score:
                                best_score = len(current)
                                best = list(current)
                            break
                    if not added:
                        break

    # Return the best result found
    if best is not None and best_score >= 3:
        # Determine D: try D=3 first if score >= 6, otherwise D=1
        if best_score >= 6:
            # Verify with D=3
            D_test = 3
            valid = True
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if not _valid_distance(best[i][0], best[i][1], best[j][0], best[j][1], D_test):
                        valid = False
                        break
                if not valid:
                    break
            if valid and _check_general_position(best, D_test):
                return _build_result(best, D_test)
            # Try D=1
            D_test = 1
            valid = True
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if not _valid_distance(best[i][0], best[i][1], best[j][0], best[j][1], D_test):
                        valid = False
                        break
                if not valid:
                    break
            if valid and _check_general_position(best, D_test):
                return _build_result(best, D_test)
        
        # Default to D=1
        return _build_result(best, 1)

    # Fallback: return simple 3-point set
    return _build_result(simple_3, 1)
