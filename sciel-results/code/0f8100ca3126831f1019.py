"""
Search for integral point sets in general position with D=1 (rational distances).
We search over integer lattice points, find the largest subset with all pairwise
distances integer, no 3 collinear, no 4 concyclic.
"""

import time
from math import isqrt, gcd
from fractions import Fraction

_START = time.time()
TIME_LIMIT = 120.0
SAFETY = 5.0

def _elapsed():
    return time.time() - _START

def _deadline_reached():
    return _elapsed() > (TIME_LIMIT - SAFETY)

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _int_dist2(p1, p2):
    """Integer squared distance between two integer points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _collinear3(p1, p2, p3):
    """Check if 3 integer points are collinear."""
    return (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]) == 0

def _concyclic4(p1, p2, p3, p4):
    """Check if 4 integer points are concyclic using exact arithmetic."""
    # Circumcircle of p1,p2,p3: find center (cx,cy) then check p4 lies on it.
    # (x-cx)^2 + (y-cy)^2 = R^2
    # 2*(x2-x1)*cx + 2*(y2-y1)*cy = x2^2+y2^2 - x1^2-y1^2
    # 2*(x3-x1)*cx + 2*(y3-y1)*cy = x3^2+y3^2 - x1^2-y1^2
    ax, ay = p1[0], p1[1]
    bx, by = p2[0], p2[1]
    cx, cy = p3[0], p3[1]
    dx, dy = p4[0], p4[1]
    
    # Linear system for center
    a1 = 2*(bx - ax)
    b1 = 2*(by - ay)
    c1 = bx*bx + by*by - ax*ax - ay*ay
    
    a2 = 2*(cx - ax)
    b2 = 2*(cy - ay)
    c2 = cx*cx + cy*cy - ax*ax - ay*ay
    
    det = a1*b2 - a2*b1
    if det == 0:
        return False  # p1,p2,p3 collinear -> degenerate
    
    # Center (cx0, cy0) = (c1*b2 - c2*b1)/det, (a1*c2 - a2*c1)/det
    # R^2 = (bx-cx0)^2 + (by-cy0)^2
    # Check: (dx-cx0)^2 + (dy-cy0)^2 == R^2
    # Multiply through by det^2 to avoid fractions:
    # (dx*det - (c1*b2-c2*b1))^2 + (dy*det - (a1*c2-a2*c1))^2 == (bx*det - (c1*b2-c2*b1))^2 + (by*det - (a1*c2-a2*c1))^2
    cx0_num = c1*b2 - c2*b1
    cy0_num = a1*c2 - a2*c1
    
    lhs = (dx*det - cx0_num)**2 + (dy*det - cy0_num)**2
    rhs = (bx*det - cx0_num)**2 + (by*det - cy0_num)**2
    return lhs == rhs

def _valid_subset(pts, indices):
    """Check all constraints for a subset of points at given indices."""
    n = len(indices)
    for i in range(n):
        for j in range(i+1, n):
            if not _is_square(_int_dist2(pts[indices[i]], pts[indices[j]])):
                return False
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[indices[i]], pts[indices[j]], pts[indices[k]]):
                    return False
    # Check no 4 concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _concyclic4(pts[indices[i]], pts[indices[j]], pts[indices[k]], pts[indices[l]]):
                        return False
    return True

def _search_small_grid(max_coord, target_size=6):
    """Search for target_size integer points in [-max_coord, max_coord]^2."""
    # Generate candidate points
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    
    # Precompute distance squares
    n = len(candidates)
    dist2 = [[0]*n for _ in range(n)]
    is_sq = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = _int_dist2(candidates[i], candidates[j])
            dist2[i][j] = dist2[j][i] = d
            s = _is_square(d)
            is_sq[i][j] = is_sq[j][i] = s
    
    # Build compatibility graph: edge if integer distance
    # Use backtracking to find largest clique with constraints
    # This is expensive, so we use a heuristic approach
    
    # Better approach: greedy + local improvement
    # Start with a seed, add points one by one
    
    # Actually, for small grids, let's try all subsets of size target_size
    # But C(1600, 6) is way too large.
    
    # Use a smarter search: iterate over all 5-tuples that are valid,
    # then try to add a 6th point.
    
    # Pre-filter: keep only points that have at least target_size-1 integer-distance neighbors
    degree = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if is_sq[i][j]:
                degree[i] += 1
                degree[j] += 1
    
    # Filter to high-degree points
    min_deg = target_size - 1
    good = [i for i in range(n) if degree[i] >= min_deg]
    
    if len(good) < target_size:
        return None
    
    # Backtracking search for clique of size target_size with constraints
    # Sort good points by degree descending
    good.sort(key=lambda i: -degree[i])
    
    best = None
    
    def backtrack(path, start):
        nonlocal best
        if _deadline_reached():
            return
        if len(path) == target_size:
            best = path[:]
            return
        remaining_needed = target_size - len(path)
        for i in range(start, len(good) - remaining_needed + 1):
            if _deadline_reached():
                return
            idx = good[i]
            # Check distance to all in path
            ok = True
            for p in path:
                if not is_sq[p][idx]:
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity with pairs in path
            ok = True
            for a in range(len(path)):
                for b in range(a+1, len(path)):
                    if _collinear3(candidates[path[a]], candidates[path[b]], candidates[idx]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check concyclicity with triples in path
            ok = True
            for a in range(len(path)):
                for b in range(a+1, len(path)):
                    for c in range(b+1, len(path)):
                        if _concyclic4(candidates[path[a]], candidates[path[b]], candidates[path[c]], candidates[idx]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            path.append(idx)
            backtrack(path, i + 1)
            path.pop()
            if best is not None:
                return
    
    backtrack([], 0)
    
    if best is not None:
        return [candidates[i] for i in best]
    return None

def _try_known_sets():
    """Try some known configurations."""
    # Try the original 5-point set
    pts5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point that works with these 5
    # Need (x,y) integer such that distance to each is integer, no collinear/concyclic
    # This is a system of 5 equations - very constrained
    # Let's just try a small search around the 5 points
    
    # Try some hand-picked 6-point sets
    candidates_6 = [
        # Various attempts
        [(0,0), (3,4), (0,7), (4,5), (7,8), (5,12)],
        [(0,0), (5,12), (12,5), (15,8), (8,15), (13,14)],
        [(0,0), (9,12), (12,9), (15,8), (8,15), (5,12)],
        [(0,0), (3,4), (12,5), (9,12), (15,8), (8,15)],
        [(0,0), (12,0), (0,9), (12,9), (5,12), (15,8)],
    ]
    
    for pts in candidates_6:
        if _valid_subset(pts, list(range(len(pts)))):
            return pts
    
    return None

def _search_grid_progressive():
    """Search with increasing grid size."""
    # First try known sets
    result = _try_known_sets()
    if result is not None:
        return result
    
    # Try small grids
    for mc in [15, 20, 25, 30, 40, 50]:
        if _deadline_reached():
            break
        result = _search_small_grid(mc, target_size=6)
        if result is not None:
            return result
    
    return None

def construct():
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]  # 5-point fallback
    
    # Try to find 6 points
    result = _search_grid_progressive()
    if result is not None and len(result) >= 6:
        best_points = result
    
    # Try to extend to 7 points if we have 6
    if len(best_points) >= 6:
        base = best_points[:6]
        # Try to find a 7th point
        for mc in [50, 75, 100]:
            if _deadline_reached():
                break
            candidates = []
            for x in range(-mc, mc + 1):
                for y in range(-mc, mc + 1):
                    candidates.append((x, y))
            
            found_7 = False
            for pt in candidates:
                if _deadline_reached():
                    break
                # Check if pt is already in base
                if pt in base:
                    continue
                # Check distance to all 6
                ok = True
                for b in base:
                    if not _is_square(_int_dist2(pt, b)):
                        ok = False
                        break
                if not ok:
                    continue
                # Check collinearity
                ok = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if _collinear3(base[i], base[j], pt):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Check concyclicity
                ok = True
                for i in range(6):
                    for j in range(i+1, 6):
                        for k in range(j+1, 6):
                            if _concyclic4(base[i], base[j], base[k], pt):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Found a 7th point!
                best_points = base + [pt]
                found_7 = True
                break
            if found_7:
                break
    
    # Format output
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
