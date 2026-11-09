import time
from math import isqrt, gcd
import random

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _validate(pts, D):
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    
    # Check distances
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                return False
                
    # Check collinearity
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
                    
    # Check concyclicity
    # Precompute x^2 + y^2
    norms = [x*x + y*y for x, y in pts]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1, n1 = pts[i][0], pts[i][1], norms[i]
                x2, y2, n2 = pts[j][0], pts[j][1], norms[j]
                x3, y3, n3 = pts[k][0], pts[k][1], norms[k]
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    x4, y4, n4 = pts[l][0], pts[l][1], norms[l]
                    
                    # Determinant of 4x4 matrix
                    # | x1 y1 n1 1 |
                    # | x2 y2 n2 1 |
                    # | x3 y3 n3 1 |
                    # | x4 y4 n4 1 |
                    # Subtract row 4 from 1,2,3
                    a1 = x1 - x4; b1 = y1 - y4; c1 = n1 - n4
                    a2 = x2 - x4; b2 = y2 - y4; c2 = n2 - n4
                    a3 = x3 - x4; b3 = y3 - y4; c3 = n3 - n4
                    
                    det = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                    if det == 0:
                        return False
    return True

def _try_add_point(pts, D, candidate, norms):
    cx, cy = candidate
    n = len(pts)
    
    # Check distances
    for i in range(n):
        x1, y1 = pts[i]
        dx = cx - x1
        dy = cy - y1
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
            
    # Check collinearity
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx * (cy - y1) - dy * (cx - x1) == 0:
                return False
                
    # Check concyclicity
    c4 = cx*cx + cy*cy
    for i in range(n):
        x1, y1, n1 = pts[i][0], pts[i][1], norms[i]
        for j in range(i + 1, n):
            x2, y2, n2 = pts[j][0], pts[j][1], norms[j]
            for k in range(j + 1, n):
                x3, y3, n3 = pts[k][0], pts[k][1], norms[k]
                
                a1 = x1 - cx; b1 = y1 - cy; c1 = n1 - c4
                a2 = x2 - cx; b2 = y2 - cy; c2 = n2 - c4
                a3 = x3 - cx; b3 = y3 - cy; c3 = n3 - c4
                
                det = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                if det == 0:
                    return False
    return True

def _generate_candidates(pts, D, max_coord):
    """Generate candidates with rational distance to all points in pts."""
    if not pts:
        return []
    candidates = set()
    # Use the first point as anchor for circle intersection logic if needed,
    # but for D=1, simple grid scan with filter is often fast enough for small N.
    # However, to be faster, we can intersect circles.
    # For now, efficient grid scan with early termination.
    
    x0, y0 = pts[0]
    # Bound the search area
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            p = (x, y)
            if p in pts:
                continue
            # Check distance to all
            valid = True
            for px, py in pts:
                dx = x - px
                dy = y - py
                if not _is_square(dx*dx + D*dy*dy):
                    valid = False
                    break
            if valid:
                candidates.add(p)
    return list(candidates)

def _greedy_search(seed, D, max_coord, time_limit):
    current = list(seed)
    norms = [x*x + y*y for x, y in current]
    best = list(current)
    
    # Sort candidates by some heuristic (e.g., distance to centroid)
    cx = sum(p[0] for p in current) / len(current)
    cy = sum(p[1] for p in current) / len(current)
    
    candidates = _generate_candidates(current, D, max_coord)
    candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    for cand in candidates:
        if time.time() > time_limit:
            break
        if _try_add_point(current, D, cand, norms):
            current.append(cand)
            norms.append(cand[0]**2 + cand[1]**2)
            if len(current) > len(best):
                best = list(current)
                if len(best) >= 12:
                    break
    return best

def construct():
    D = 1
    # Known 5-point set
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if not _validate(seed5, D):
        # Fallback to a trivial 3-point set if seed is invalid (should not happen)
        seed5 = [(0,0), (3,0), (0,4)]
        
    best_pts = list(seed5)
    time_limit = _DEADLINE
    
    # Phase 1: Greedy growth from seed5
    for mc in [60, 80, 100, 120, 150, 200]:
        if time.time() > time_limit:
            break
        grown = _greedy_search(seed5, D, mc, time_limit)
        if len(grown) > len(best_pts):
            best_pts = grown
    
    # Phase 2: Try other seeds
    other_seeds = [
        [(0, 0), (3, 0), (0, 4)],
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (8, 0), (0, 15)],
        [(0, 0), (7, 0), (0, 24)],
        [(0, 0), (20, 0), (0, 21)],
        [(0, 0), (9, 0), (0, 12)],
        [(0, 0), (12, 0), (0, 16)],
        [(0, 0), (28, 0), (0, 45)],
        [(0, 0), (11, 0), (0, 60)],
        [(0, 0), (9, 0), (0, 40)],
        [(0, 0), (40, 0), (0, 9)],
        [(0, 0), (19, 0), (0, 180)],
        [(0, 0), (31, 0), (0, 18)],
        [(0, 0), (18, 0), (0, 31)],
        [(0, 0), (65, 0), (0, 72)],
        [(0, 0), (16, 0), (0, 63)],
    ]
    
    for seed in other_seeds:
        if time.time() > time_limit:
            break
        if _validate(seed, D):
            for mc in [50, 70, 90, 110]:
                if time.time() > time_limit:
                    break
                grown = _greedy_search(seed, D, mc, time_limit)
                if len(grown) > len(best_pts):
                    best_pts = grown
    
    # Phase 3: Local search / Swap moves on best_pts
    # Try to remove one point and add two new ones
    if len(best_pts) < 12 and time.time() < time_limit:
        current = list(best_pts)
        improved = False
        for idx in range(len(current)):
            if time.time() > time_limit:
                break
            removed = current.pop(idx)
            # Try to add 2 points
            candidates = _generate_candidates(current, D, 100)
            added = []
            for cand in candidates:
                if cand == removed:
                    continue
                if _try_add_point(current, D, cand, [x*x+y*y for x,y in current]):
                    current.append(cand)
                    added.append(cand)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                if len(current) > len(best_pts):
                    best_pts = list(current)
                    improved = True
            # Restore
            current.pop()
            if added:
                for _ in added:
                    current.pop()
            current.insert(idx, removed)
            if improved:
                break
                
    # Final validation
    if not _validate(best_pts, D):
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in seed5]}
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
