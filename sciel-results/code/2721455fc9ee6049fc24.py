"""Integral point sets in general position - search for maximum size."""
import time
import random
from math import isqrt
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 145.0  # 145s safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = pts[k]
                if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    sqs = [(x*x + y*y) for x, y in pts]
    
    for i in range(n):
        x1, y1 = pts[i]
        c1 = sqs[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            c2 = sqs[j]
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                c3 = sqs[k]
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    x4, y4 = pts[l]
                    c4 = sqs[l]
                    a1, b1, c1d = x1-x4, y1-y4, c1-c4
                    a2, b2, c2d = x2-x4, y2-y4, c2-c4
                    a3, b3, c3d = x3-x4, y3-y4, c3-c4
                    det = a1 * (b2 * c3d - c2d * b3) - b1 * (a2 * c3d - c2d * a3) + c1d * (a2 * b3 - b2 * a3)
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                return False
    return True

def _validate(pts, D):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _all_distances_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

def _try_add_point(pts, D, candidate):
    """Check if adding candidate to pts keeps validity."""
    for p in pts:
        dx = candidate[0] - p[0]
        dy = candidate[1] - p[1]
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
    
    n = len(pts)
    x4, y4 = candidate
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx * (y4 - y1) - dy * (x4 - x1) == 0:
                return False
            
    c4 = x4*x4 + y4*y4
    for i in range(n):
        x1, y1 = pts[i]
        c1 = x1*x1 + y1*y1
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            c2 = x2*x2 + y2*y2
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                c3 = x3*x3 + y3*y3
                a1, b1, c1d = x1-x4, y1-y4, c1-c4
                a2, b2, c2d = x2-x4, y2-y4, c2-c4
                a3, b3, c3d = x3-x4, y3-y4, c3-c4
                det = a1 * (b2 * c3d - c2d * b3) - b1 * (a2 * c3d - c2d * a3) + c1d * (a2 * b3 - b2 * a3)
                if det == 0:
                    return False
    return True

def _search_with_backtracking(D, seed_pts, max_coord, time_limit, rng, max_backtrack=3):
    """Search with limited backtracking to escape local optima."""
    current = list(seed_pts)
    best = list(current)
    
    # Generate all candidates in range
    all_candidates = []
    existing_set = set(current)
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in existing_set:
                all_candidates.append((x, y))
    
    # Sort candidates by distance to centroid
    if current:
        cx = sum(p[0] for p in current) / len(current)
        cy = sum(p[1] for p in current) / len(current)
        all_candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    else:
        all_candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    # For random restart, shuffle
    if rng is not None:
        rng.shuffle(all_candidates)
    
    # Greedy with backtracking
    def _greedy_from(start_idx, current_pts, best_pts, backtrack_depth):
        nonlocal best
        if time.time() > time_limit:
            return best_pts
        
        if len(current_pts) > len(best_pts):
            best_pts = list(current_pts)
            best = list(best_pts)
        
        if len(best_pts) >= 12:
            return best_pts
        
        for idx in range(start_idx, len(all_candidates)):
            if time.time() > time_limit:
                break
            
            candidate = all_candidates[idx]
            if _try_add_point(current_pts, D, candidate):
                new_pts = current_pts + [candidate]
                # Continue greedily from next index
                result = _greedy_from(idx + 1, new_pts, best_pts, backtrack_depth)
                if len(result) > len(best_pts):
                    best_pts = result
                    best = list(best_pts)
                
                # Try backtracking: skip this candidate and try others
                if backtrack_depth > 0 and len(current_pts) < 10:
                    # Try a few alternative candidates at this level
                    for alt_idx in range(idx + 1, min(idx + 5, len(all_candidates))):
                        if time.time() > time_limit:
                            break
                        alt_candidate = all_candidates[alt_idx]
                        if alt_candidate not in current_pts and _try_add_point(current_pts, D, alt_candidate):
                            alt_pts = current_pts + [alt_candidate]
                            alt_result = _greedy_from(alt_idx + 1, alt_pts, best_pts, backtrack_depth - 1)
                            if len(alt_result) > len(best_pts):
                                best_pts = alt_result
                                best = list(best_pts)
        
        return best_pts
    
    _greedy_from(0, current, best, max_backtrack)
    return best

def construct():
    D = 1
    
    # Multiple seed sets
    seeds = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],  # Known 5-point
        [(0, 0), (3, 4), (0, 5)],
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (9, 12), (0, 15)],
        [(0, 0), (7, 24), (24, 7)],
    ]
    
    best_pts = None
    best_D = D
    time_limit = _DEADLINE - 5.0
    
    # First, validate seeds and find the best starting point
    valid_seeds = []
    for seed in seeds:
        if _validate(seed, D):
            valid_seeds.append(seed)
    
    if not valid_seeds:
        # Fallback to a simple valid set
        valid_seeds = [(0, 0), (3, 4), (0, 5)]
        D = 1
    
    # Start with the largest valid seed
    valid_seeds.sort(key=len, reverse=True)
    best_seed = valid_seeds[0]
    best_pts = list(best_seed)
    
    # Try search with backtracking from each valid seed
    rng = random.Random(42)
    
    for seed in valid_seeds:
        if time.time() > time_limit:
            break
        
        # Try different coordinate ranges
        for max_coord in [60, 100, 150]:
            if time.time() > time_limit:
                break
            
            # Deterministic search
            grown = _search_with_backtracking(D, seed, max_coord, time_limit, rng=None, max_backtrack=2)
            if best_pts is None or len(grown) > len(best_pts):
                best_pts = grown
                if len(best_pts) >= 12:
                    break
            
            # Random restarts
            for r in range(3):
                if time.time() > time_limit:
                    break
                grown = _search_with_backtracking(D, seed, max_coord, time_limit, rng=rng, max_backtrack=2)
                if best_pts is None or len(grown) > len(best_pts):
                    best_pts = grown
                    if len(best_pts) >= 12:
                        break
    
    # Try with D=2 as well
    D2 = 2
    seed5_D2 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if _validate(seed5_D2, D2):
        for max_coord in [60, 100]:
            if time.time() > time_limit:
                break
            grown = _search_with_backtracking(D2, seed5_D2, max_coord, time_limit, rng=None, max_backtrack=2)
            if best_pts is None or len(grown) > len(best_pts):
                best_pts = grown
                best_D = D2
                if len(best_pts) >= 12:
                    break
            for r in range(2):
                if time.time() > time_limit:
                    break
                grown = _search_with_backtracking(D2, seed5_D2, max_coord, time_limit, rng=rng, max_backtrack=2)
                if best_pts is None or len(grown) > len(best_pts):
                    best_pts = grown
                    best_D = D2
                    if len(best_pts) >= 12:
                        break
    
    if best_pts is None:
        best_pts = [(0, 0), (3, 4), (0, 5)]
        best_D = 1
    
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
