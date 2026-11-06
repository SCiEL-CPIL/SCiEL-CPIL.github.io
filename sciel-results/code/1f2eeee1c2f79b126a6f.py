"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0

    def is_squarefree(n):
        if n <= 1:
            return True
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                return False
            p += 1 if p == 2 else 2
        return True

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_valid(points, D):
        n = len(points)
        if n < 3:
            return False
        
        seen = set()
        for (x, y) in points:
            key = (x, y)
            if key in seen:
                return False
            seen.add(key)

        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + D * dy * dy
                if d2 < 0:
                    return False
                num = d2.numerator
                den = d2.denominator
                if not is_perfect_square(num) or not is_perfect_square(den):
                    return False

        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                        return False

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        a1 = x1 * x1 + D * y1 * y1
                        a2 = x2 * x2 + D * y2 * y2
                        a3 = x3 * x3 + D * y3 * y3
                        a4 = x4 * x4 + D * y4 * y4
                        
                        det = (x1 * (y2 * (a3 - a4) - y3 * (a2 - a4) + y4 * (a2 - a3)) -
                               y1 * (x2 * (a3 - a4) - x3 * (a2 - a4) + x4 * (a2 - a3)) +
                               a1 * (x2 * y3 - x3 * y2 - x2 * y4 + x4 * y2 + x3 * y4 - x4 * y3) -
                               (x2 * y3 * a4 - x2 * y4 * a3 - x3 * y2 * a4 + x3 * y4 * a2 + x4 * y2 * a3 - x4 * y3 * a2))
                        if det == 0:
                            return False
        return True

    def search_6_points(D, limit):
        candidates = []
        for x in range(-limit, limit + 1):
            for y in range(-limit, limit + 1):
                candidates.append((Fraction(x), Fraction(y)))
        
        n = len(candidates)
        # Precompute adjacency: can[i][j] is True if distance is rational
        # For D=1 and integer coords, distance is integer iff dx^2+dy^2 is square.
        # We store indices.
        can = [[False] * n for _ in range(n)]
        for i in range(n):
            x1, y1 = candidates[i]
            for j in range(i + 1, n):
                x2, y2 = candidates[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + D * dy * dy
                if d2 >= 0:
                    num = d2.numerator
                    den = d2.denominator
                    if is_perfect_square(num) and is_perfect_square(den):
                        can[i][j] = True
                        can[j][i] = True

        best = None
        current = []
        
        def backtrack(idx):
            nonlocal best
            if time.time() > deadline:
                return
            if len(current) == 6:
                if check_valid(current, D):
                    best = list(current)
                return
            
            # Pruning
            if n - idx < 6 - len(current):
                return

            for i in range(idx, n):
                if not can[current[-1]][i] if current else True:
                    # If current is not empty, i must be connected to the last added point
                    # Actually, in clique search, i must be connected to ALL points in current.
                    pass
                
                # Check connectivity to all in current
                valid = True
                for k in current:
                    if not can[k][i]:
                        valid = False
                        break
                if not valid:
                    continue
                
                current.append(i)
                backtrack(i + 1)
                current.pop()
                if best is not None:
                    return

        backtrack(0)
        return best

    # Try to find 6 points with D=1
    # Start with small limits
    found = None
    for limit in [15, 20, 25]:
        if time.time() > deadline - 10:
            break
        result = search_6_points(1, limit)
        if result:
            found = result
            break

    if found:
        points_out = [[[int(x.numerator), int(x.denominator)], [int(y.numerator), int(y.denominator)]] for x, y in found]
        return {'D': 1, 'points': points_out}
    else:
        # Fallback to 5 points
        points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        D_5 = 1
        fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
        points_out = [[[int(x.numerator), int(x.denominator)], [int(y.numerator), int(y.denominator)]] for x, y in fracs_5]
        return {'D': D_5, 'points': points_out}

# EVOLVE-BLOCK-END
