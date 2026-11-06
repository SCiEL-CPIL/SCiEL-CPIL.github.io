"""Optimized search for integral-distance point sets with rational coordinates."""
from fractions import Fraction
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True


def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _dist2_is_rational_sq(p1, p2, D):
    """Check if squared distance between two points is a perfect square rational."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    return _is_perfect_square(num) and _is_perfect_square(den)


def _check_pairwise(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist2_is_rational_sq(pts[i], pts[j], D):
                return False
    return True


def _check_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                   (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                    return False
    return True


def _check_concyclic(pts, D):
    """Check no four points lie on a conic x^2 + D*y^2 + Ax + By + C = 0."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1 * x1 + D * y1 * y1
                    a2 = x2 * x2 + D * y2 * y2
                    a3 = x3 * x3 + D * y3 * y3
                    a4 = x4 * x4 + D * y4 * y4
                    det = (x1 * (y2 * (a3 - a4) - y3 * (a2 - a4) + y4 * (a2 - a3)) -
                           y1 * (x2 * (a3 - a4) - x3 * (a2 - a4) + x4 * (a2 - a3)) +
                           a1 * (x2 * y3 - x3 * y2 - x2 * y4 + x4 * y2 + x3 * y4 - x4 * y3) -
                           (x2 * y3 * a4 - x2 * y4 * a3 - x3 * y2 * a4 + x3 * y4 * a2 +
                            x4 * y2 * a3 - x4 * y3 * a2))
                    if det == 0:
                        return False
    return True


def _check_valid(pts, D):
    n = len(pts)
    if n < 3:
        return False
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    if not _check_pairwise(pts, D):
        return False
    if not _check_collinear(pts):
        return False
    if n >= 4 and not _check_concyclic(pts, D):
        return False
    return True


def _to_output(pts):
    return [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]


def _generate_pythagorean_candidates(limit, D):
    """Generate candidates using Pythagorean triples and rational parametrization."""
    candidates = set()
    
    # Add origin
    candidates.add((Fraction(0), Fraction(0)))
    
    # Generate points using Pythagorean triples
    # For D=1: (a, b) where a² + b² is a square
    # Generalize: (x, y) where x² + D*y² is a square rational
    
    for a in range(1, limit + 1):
        for b in range(1, limit + 1):
            # Check if a² + D*b² is a perfect square
            val = a * a + D * b * b
            if _is_perfect_square(val):
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        candidates.add((Fraction(sx * a), Fraction(sy * b)))
    
    # Rational parametrization of the conic x² + D*y² = 1
    # Using rational point (1,0) and slope t:
    # Line y = t(x-1), substitute: x² + D*t²(x-1)² = 1
    # (1 + D*t²)x² - 2*D*t²*x + D*t² - 1 = 0
    # One root is x=1, other root: x = (D*t²-1)/(1+D*t²)
    # y = t(x-1) = t((D*t²-1)/(1+D*t²) - 1) = t((D*t²-1-1-D*t²)/(1+D*t²)) = -2t/(1+D*t²)
    
    for t_num in range(-limit, limit + 1):
        for t_den in range(1, limit + 1):
            if math.gcd(t_num, t_den) != 1:
                continue
            t = Fraction(t_num, t_den)
            Dt2 = D * t * t
            denom = 1 + Dt2
            if denom == 0:
                continue
            # x = (D*t²-1)/(1+D*t²), y = -2t/(1+D*t²)
            x = (Dt2 - 1) / denom
            y = -2 * t / denom
            
            # These are rational points on the conic
            # Scale to get integer coordinates if possible
            # The point (x, y) has rational coordinates
            # Distance from origin: sqrt(x² + D*y²) = 1 (on the conic)
            # But we need points where pairwise distances are rational
            # So we need to scale these points appropriately
            
            # Add the point and its reflections
            candidates.add((x, y))
            candidates.add((-x, y))
            candidates.add((x, -y))
            candidates.add((-x, -y))
            
            # Also try scaling by various factors to get integer coordinates
            for scale_num in range(1, min(limit, 10) + 1):
                sx = x * scale_num
                sy = y * scale_num
                if sx.denominator == 1 and sy.denominator == 1:
                    candidates.add((sx, sy))
                    candidates.add((-sx, sy))
                    candidates.add((sx, -sy))
                    candidates.add((-sx, -sy))
    
    return list(candidates)


def _search_backtracking(candidates, D, target_n, deadline, max_nodes=500000):
    """Backtracking search with precomputed compatibility and better pruning."""
    n_cand = len(candidates)
    if n_cand < target_n:
        return None
    
    # Precompute pairwise compatibility
    compat = [[True] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            dx = candidates[i][0] - candidates[j][0]
            dy = candidates[i][1] - candidates[j][1]
            d2 = dx * dx + D * dy * dy
            if d2 < 0 or not (_is_perfect_square(d2.numerator) and _is_perfect_square(d2.denominator)):
                compat[i][j] = compat[j][i] = False
    
    # Sort candidates by number of compatible neighbors (descending) for better pruning
    deg = [0] * n_cand
    for i in range(n_cand):
        for j in range(n_cand):
            if i != j and compat[i][j]:
                deg[i] += 1
    
    # Create ordering: high degree first
    order = sorted(range(n_cand), key=lambda i: -deg[i])
    pos = [0] * n_cand  # pos[i] = position of candidate i in order
    for p, i in enumerate(order):
        pos[i] = p
    
    # Rebuild compat in new order
    compat2 = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(n_cand):
            if i != j:
                compat2[i][j] = compat[order[i]][order[j]]
    
    result_box = [None]
    node_count = [0]
    
    def bt(current, current_set, idx):
        if time.time() > deadline or node_count[0] > max_nodes:
            return False
        if len(current) >= target_n:
            if _check_concyclic(current, D):
                result_box[0] = list(current)
                return True
            return False
        
        # Try to add candidates from idx onwards
        for i in range(idx, n_cand):
            node_count[0] += 1
            if time.time() > deadline or node_count[0] > max_nodes:
                return False
            
            # Check compatibility with all points in current
            ok = True
            for j in current_set:
                if not compat2[i][j]:
                    ok = False
                    break
            if not ok:
                continue
            
            cand = candidates[order[i]]
            new_cur = current + [cand]
            
            # Check collinearity
            if not _check_collinear(new_cur):
                continue
            
            new_set = current_set | {i}
            if bt(new_cur, new_set, i + 1):
                return True
        
        return False
    
    if bt([], set(), 0):
        return result_box[0]
    return None


def _try_extend(pts, D, candidates, deadline, max_size=12):
    """Try to extend a valid set with more points."""
    best = list(pts)
    n = len(best)
    
    # Try adding one point at a time
    for _ in range(max_size - n):
        if time.time() > deadline:
            break
        added = False
        for cand in candidates:
            if cand in best:
                continue
            new_pts = best + [cand]
            if _check_valid(new_pts, D):
                best = new_pts
                added = True
                break
        if not added:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Fallback: known 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D_5 = 1
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    best_result = {'D': D_5, 'points': _to_output(fracs_5), 'n': 5}
    
    def remaining():
        return deadline - time.time()
    
    # Strategy 1: Search on integer grid for D=1 with larger limits
    D = 1
    for limit in [10, 12, 15, 18, 20]:
        if remaining() < 8:
            break
        
        candidates = _generate_pythagorean_candidates(limit, D)
        if len(candidates) < 6:
            continue
        
        # Shuffle for diversity
        random.seed(42 + limit)
        random.shuffle(candidates)
        
        # Limit candidate set size for performance
        max_cands = 300
        if len(candidates) > max_cands:
            candidates = candidates[:max_cands]
        
        result = _search_backtracking(candidates, D, 6, deadline, max_nodes=300000)
        if result:
            best_result = {'D': D, 'points': _to_output(result), 'n': len(result)}
            
            # Try to extend
            if remaining() > 8:
                extended = _try_extend(result, D, candidates, deadline)
                if len(extended) > best_result['n']:
                    best_result = {'D': D, 'points': _to_output(extended), 'n': len(extended)}
            
            if best_result['n'] >= 7:
                break
    
    # Strategy 2: Try other D values
    if remaining() > 15:
        for D in [2, 3, 5, 7, 11, 13]:
            if remaining() < 8:
                break
            if not _is_squarefree(D):
                continue
            
            for limit in [8, 10, 12]:
                if remaining() < 6:
                    break
                
                candidates = _generate_pythagorean_candidates(limit, D)
                if len(candidates) < 6:
                    continue
                
                random.seed(D * 100 + limit)
                random.shuffle(candidates)
                
                max_cands = 200
                if len(candidates) > max_cands:
                    candidates = candidates[:max_cands]
                
                result = _search_backtracking(candidates, D, 6, deadline, max_nodes=150000)
                if result and len(result) > best_result['n']:
                    best_result = {'D': D, 'points': _to_output(result), 'n': len(result)}
                    
                    if remaining() > 6:
                        extended = _try_extend(result, D, candidates, deadline)
                        if len(extended) > best_result['n']:
                            best_result = {'D': D, 'points': _to_output(extended), 'n': len(extended)}
                
                if best_result['n'] >= 7:
                    break
    
    # Strategy 3: Try known configurations
    if remaining() > 5:
        known_configs = [
            (1, [(0,0), (3,0), (0,4), (4,3), (-3,4), (0,-5)]),
            (1, [(0,0), (5,0), (0,12), (12,5), (-5,12), (0,-13)]),
            (1, [(0,0), (7,0), (0,24), (24,7), (-7,24), (0,-25)]),
            (1, [(0,0), (9,0), (0,40), (40,9), (-9,40), (0,-41)]),
            (1, [(0,0), (4,0), (0,3), (4,3), (-4,3), (0,-5)]),
            (1, [(0,0), (5,0), (0,4), (5,4), (-5,4), (0,-3)]),
            (1, [(0,0), (8,0), (0,15), (15,8), (-8,15), (0,-16)]),
            (1, [(0,0), (12,0), (0,35), (35,12), (-12,35), (0,-36)]),
            (1, [(0,0), (15,0), (0,20), (20,15), (-15,20), (0,-25)]),
            (1, [(0,0), (20,0), (0,21), (21,20), (-20,21), (0,-29)]),
            (2, [(0,0), (1,0), (0,1), (1,1), (-1,1), (1,-1)]),
            (2, [(0,0), (2,0), (0,1), (2,1), (-2,1), (2,-1)]),
            (3, [(0,0), (1,0), (0,1), (1,1), (-1,1), (1,-1)]),
            (5, [(0,0), (1,0), (0,1), (1,1), (-1,1), (1,-1)]),
        ]
        for D, pts in known_configs:
            if not _is_squarefree(D):
                continue
            fracs = [(Fraction(x), Fraction(y)) for x, y in pts]
            if _check_valid(fracs, D) and len(fracs) > best_result['n']:
                best_result = {'D': D, 'points': _to_output(fracs), 'n': len(fracs)}
    
    # Strategy 4: Final extension attempt
    if remaining() > 10 and best_result['n'] < 8:
        D_cur = best_result['D']
        pts_cur = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) for p in best_result['points']]
        
        # Generate broader candidate set
        limit_ext = 25
        candidates_ext = _generate_pythagorean_candidates(limit_ext, D_cur)
        extended = _try_extend(pts_cur, D_cur, candidates_ext, deadline)
        if len(extended) > best_result['n']:
            best_result = {'D': D_cur, 'points': _to_output(extended), 'n': len(extended)}
        
        # Try with different D values
        if remaining() > 5:
            for D_try in [1, 2, 3, 5]:
                if remaining() < 3:
                    break
                candidates_try = _generate_pythagorean_candidates(20, D_try)
                if len(candidates_try) < 6:
                    continue
                # Try to find a new set from scratch
                result_try = _search_backtracking(candidates_try[:200], D_try, 6, deadline, max_nodes=100000)
                if result_try and len(result_try) > best_result['n']:
                    extended_try = _try_extend(result_try, D_try, candidates_try, deadline)
                    if len(extended_try) > best_result['n']:
                        best_result = {'D': D_try, 'points': _to_output(extended_try), 'n': len(extended_try)}
    
    del best_result['n']
    return best_result

# EVOLVE-BLOCK-END
