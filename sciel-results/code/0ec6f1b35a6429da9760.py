"""Rewritten program: construct a larger integral-point set in general position."""
from fractions import Fraction
from math import isqrt
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a set of points (x, y*sqrt(D)) with rational coordinates such that:
    - All pairwise distances are rational.
    - No three points are collinear.
    - No four points are concyclic.
    - D is squarefree, D <= 10^12.
    - |numerators|, denominators <= 10^40.
    Maximize the number of points.
    
    Strategy: Use D=1 (ordinary integer lattice) and search for a large set of
    integer points with all pairwise distances integer, no three collinear, no four
    concyclic. We use a systematic greedy construction with multiple starting points.
    """
    
    def dist_sq(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2
    
    def is_integer_dist(p, q):
        d2 = dist_sq(p, q)
        r = isqrt(d2)
        return r * r == d2
    
    def are_collinear(p, q, r):
        # Cross product
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0
    
    def are_concyclic(p, q, r, s):
        # Four points are concyclic iff the determinant is zero
        # |x^2+y^2, x, y, 1| for each point
        def row(pt):
            x, y = pt
            return [x*x + y*y, x, y, 1]
        M = [row(p), row(q), row(r), row(s)]
        # Compute determinant via Gaussian elimination over rationals
        n = 4
        M = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
        det = Fraction(1)
        for i in range(n):
            # Find pivot
            pivot_row = -1
            for r in range(i, n):
                if M[r][i] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return True  # determinant is 0
            if pivot_row != i:
                M[i], M[pivot_row] = M[pivot_row], M[i]
                det = -det
            det *= M[i][i]
            # Eliminate below
            for r in range(i+1, n):
                if M[r][i] != 0:
                    factor = M[r][i] / M[i][i]
                    for c in range(i, n):
                        M[r][c] -= factor * M[i][c]
        return det == 0
    
    def valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not is_integer_dist(points[i], points[j]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True
    
    def try_add_point(points, candidate):
        """Check if candidate can be added to points."""
        # Check integer distance to all existing
        for p in points:
            if not is_integer_dist(p, candidate):
                return False
        # Check no three collinear
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if are_collinear(points[i], points[j], candidate):
                    return False
        # Check no four concyclic
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if are_concyclic(points[i], points[j], points[k], candidate):
                        return False
        return True
    
    def find_candidates_from_pair(p, q, max_radius=200):
        """Find lattice points at integer distance from both p and q."""
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        d2 = dx*dx + dy*dy
        d = isqrt(d2)
        if d * d != d2 or d == 0:
            return []
        
        candidates = set()
        px, py = p[0], p[1]
        qx, qy = q[0], q[1]
        for r in range(1, max_radius + 1):
            for ax in range(-r, r + 1):
                rem = r*r - ax*ax
                if rem < 0:
                    continue
                by = isqrt(rem)
                if by * by != rem:
                    continue
                for by_val in ([by] if by == 0 else [by, -by]):
                    cand = (px + ax, py + by_val)
                    if cand == p or cand == q:
                        continue
                    if is_integer_dist(q, cand):
                        candidates.add(cand)
        return list(candidates)
    
    def greedy_build(initial_points, max_points=8, search_radius=100):
        """Greedily add points to the set."""
        points = list(initial_points)
        used = set(points)
        
        for _ in range(max_points - len(points)):
            added = False
            all_candidates = set()
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    cands = find_candidates_from_pair(points[i], points[j], max_radius=search_radius)
                    all_candidates.update(cands)
            
            for cand in sorted(all_candidates):
                if cand in used:
                    continue
                if try_add_point(points, cand):
                    points.append(cand)
                    used.add(cand)
                    added = True
                    break
            
            if not added:
                break
        
        return points
    
    # Known 5-point base set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to build from base5
    best = greedy_build(base5, max_points=8, search_radius=100)
    
    # Try another known configuration
    base5b = [(0, 0), (4, 3), (5, 12), (16, 63), (0, 63)]
    best2 = greedy_build(base5b, max_points=8, search_radius=100)
    if len(best2) > len(best):
        best = best2
    
    # Verify
    if not valid_set(best):
        best = list(base5)
    
    # Convert to required format
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
