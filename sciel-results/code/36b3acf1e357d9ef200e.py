"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from math import isqrt
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 140.0

    def is_square_frac(f):
        if f < 0:
            return False, None
        n, d = f.numerator, f.denominator
        if n == 0:
            return True, Fraction(0, 1)
        rn = isqrt(n)
        rd = isqrt(d)
        if rn * rn == n and rd * rd == d:
            return True, Fraction(rn, rd)
        return False, None

    def valid_set(xs):
        """xs: list of Fraction x-coordinates on parabola y=x^2.
        Check: all pairwise distances rational, no four concyclic.
        (No three collinear is automatic for distinct points on a parabola.)"""
        n = len(xs)
        for i in range(n):
            for j in range(i+1, n):
                s = xs[i] + xs[j]
                val = 1 + s*s
                ok, _ = is_square_frac(val)
                if not ok:
                    return False
        # No four concyclic: x1+x2+x3+x4 != 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if xs[i] + xs[j] + xs[k] + xs[l] == 0:
                            return False
        return True

    # Generate candidate x-values: x = (u - 1/u)/2 for rational u
    # This ensures 1+x^2 is a perfect square (since x^2+1 = (u+1/u)^2/4)
    # Wait: 1 + ((u-1/u)/2)^2 = 1 + (u^2-2+1/u^2)/4 = (u^2+2+1/u^2)/4 = ((u+1/u)/2)^2. Yes!
    
    # So if we pick x-values from this family, distance to origin is rational.
    # But we need ALL pair distances rational, not just distance to origin.
    
    # For two such x-values x_a = (u_a - 1/u_a)/2, x_b = (u_b - 1/u_b)/2:
    # x_a + x_b = (u_a - 1/u_a + u_b - 1/u_b)/2
    # Need 1 + (x_a+x_b)^2 to be a perfect square.
    
    # Let's just generate a large pool and search.
    
    candidates = set()
    candidates.add(Fraction(0, 1))
    
    max_pq = 40
    for p in range(-max_pq, max_pq+1):
        for q in range(1, max_pq+1):
            if p == 0:
                continue
            # Reduce fraction
            from math import gcd
            g = gcd(abs(p), q)
            pp, qq = p//g, q//g
            u = Fraction(pp, qq)
            if u in (Fraction(1,1), Fraction(-1,1)):
                continue
            x_val = (u - Fraction(1, u)) / 2
            candidates.add(x_val)
    
    # Also try some direct small rational x values and check
    for p in range(-50, 51):
        for q in range(1, 15):
            from math import gcd
            g = gcd(abs(p), q) if p != 0 else q
            pp, qq = p//g, q//g
            x_val = Fraction(pp, qq)
            # Check if 1+x^2 is a perfect square
            val = 1 + x_val*x_val
            ok, _ = is_square_frac(val)
            if ok:
                candidates.add(x_val)
    
    candidates = list(candidates)
    
    # Sort by height
    def height(f):
        return abs(f.numerator) + abs(f.denominator)
    candidates.sort(key=height)
    candidates = candidates[:800]
    
    n_cand = len(candidates)
    
    # Precompute pair validity
    pair_ok = [[False]*n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            s = candidates[i] + candidates[j]
            val = 1 + s*s
            ok, _ = is_square_frac(val)
            pair_ok[i][j] = ok
            pair_ok[j][i] = ok
    
    # Greedy search with randomization
    random.seed(12345)
    best_set = None
    best_size = 0
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(n_cand))
        random.shuffle(order)
        
        current = []
        for idx in order:
            if time.time() > deadline:
                break
            ok = True
            for j in current:
                if not pair_ok[idx][j]:
                    ok = False
                    break
            if not ok:
                continue
            # Check no four concyclic
            four_ok = True
            for a in current:
                for b in current:
                    for c in current:
                        if a >= b or b >= c:
                            continue
                        if candidates[idx] + candidates[a] + candidates[b] + candidates[c] == 0:
                            four_ok = False
                            break
                    if not four_ok:
                        break
                if not four_ok:
                    break
            if not four_ok:
                continue
            
            current.append(idx)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 8:
                break
    
    if best_set is None or len(best_set) < 3:
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    # Final verification
    xs = [candidates[i] for i in best_set]
    if not valid_set(xs):
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    result_points = []
    for idx in best_set:
        x = candidates[idx]
        y = x * x
        result_points.append([list(x.as_integer_ratio()), list(y.as_integer_ratio())])
    
    return {'D': 1, 'points': result_points}

# EVOLVE-BLOCK-END
