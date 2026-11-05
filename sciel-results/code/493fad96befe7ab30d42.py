"""Rewritten program for integral point sets in general position.
Searches for larger configurations than the 5-point seed.
"""
import time
import random
from math import isqrt, gcd
from fractions import Fraction
from itertools import combinations

# Deadline guard
T0 = time.time()
TIME_LIMIT = 115  # seconds, safety margin before 120s

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(p, q, D):
    """Squared distance between p=(x1,y1) and q=(x2,y2) in the metric
    where actual point is (x, y*sqrt(D)).
    Returns Fraction."""
    x1, y1 = p
    x2, y2 = q
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _check_rational_distances(points, D):
    """Check all pairwise squared distances are perfect squares of rationals."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            sd = _sqdist(points[i], points[j], D)
            if sd < 0:
                return False
            num, den = sd.numerator, sd.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _check_no_three_collinear(points, D):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a11 = x2 - x1
        a12 = y2 - y1
        a21 = x3 - x1
        a22 = y3 - y1
        b1 = -((x2*x2 - x1*x1) + D*(y2*y2 - y1*y1))
        b2 = -((x3*x3 - x1*x1) + D*(y3*y3 - y1*y1))
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            continue
        
        A = (b1 * a22 - b2 * a12) / det
        B = (a11 * b2 - a21 * b1) / det
        C = -(x1*x1 + D*y1*y1 + A*x1 + B*y1)
        
        val = x4*x4 + D*y4*y4 + A*x4 + B*y4 + C
        if val == 0:
            return False
    return True

def _is_valid(points, D):
    """Full validation."""
    if not _check_rational_distances(points, D):
        return False
    if not _check_no_three_collinear(points, D):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _format_result(points, D):
    """Convert points list of Fraction tuples to the required format."""
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    return {'D': D, 'points': result_points}

def _has_rational_dist(p, q, D):
    """Check if squared distance is a rational square."""
    sd = _sqdist(p, q, D)
    if sd < 0:
        return False
    num, den = sd.numerator, sd.denominator
    return _is_square(num) and _is_square(den)

def _search_clique(candidates, D, time_budget=10):
    """Find largest subset with all pairwise rational distances, then check other constraints."""
    n = len(candidates)
    if n < 3:
        return None, 0
    
    # Build adjacency: edge if rational distance
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _has_rational_dist(candidates[i], candidates[j], D):
                adj[i].add(j)
                adj[j].add(i)
    
    # Find max clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() - T0 > TIME_LIMIT - 5:
            return
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if len(R) + len(P) <= len(best_clique):
            return
        
        # Choose pivot from X | P
        union = X | P
        if union:
            pivot = max(union, key=lambda v: len(adj[v] & P))
            candidates_iter = P - adj[pivot]
        else:
            candidates_iter = P
        
        for v in list(candidates_iter):
            R.add(v)
            bron_kerbosch(R, P & adj[v], X & adj[v])
            R.discard(v)
            P.discard(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    if not best_clique:
        return None, 0
    
    pts = [candidates[i] for i in best_clique]
    
    # Check other constraints
    if _check_no_three_collinear(pts, D) and _check_no_four_concyclic(pts, D):
        return pts, len(pts)
    
    # If full clique fails, try subsets
    for size in range(len(best_clique) - 1, 2, -1):
        for subset in combinations(best_clique, size):
            pts = [candidates[i] for i in subset]
            if _check_no_three_collinear(pts, D) and _check_no_four_concyclic(pts, D):
                return pts, size
        break
    
    return None, 0

def construct():
    """Main entry point. Searches for the largest valid configuration."""
    global T0
    T0 = time.time()
    
    best_points = None
    best_n = 0
    best_D = 1
    
    # First, verify the seed
    seed_points = [(Fraction(0), Fraction(0)), 
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    
    if _is_valid(seed_points, 1):
        best_points = seed_points
        best_n = 5
        best_D = 1
    
    # Search D=1 with integer coordinates
    if time.time() - T0 < TIME_LIMIT - 30:
        candidates = [(Fraction(x), Fraction(y)) for x in range(-25, 26) for y in range(-25, 26)]
        pts, n = _search_clique(candidates, 1)
        if pts and n > best_n:
            best_points = pts
            best_n = n
            best_D = 1
    
    # Search D=1 with rational coordinates (denominators 1-4)
    if time.time() - T0 < TIME_LIMIT - 20:
        candidates = []
        seen = set()
        for x_num in range(-40, 41):
            for x_den in [1, 2, 3, 4]:
                for y_num in range(-40, 41):
                    for y_den in [1, 2, 3, 4]:
                        pt = (Fraction(x_num, x_den), Fraction(y_num, y_den))
                        if pt not in seen:
                            seen.add(pt)
                            candidates.append(pt)
        
        # Limit to reasonable size
        if len(candidates) > 800:
            candidates = candidates[:800]
        
        pts, n = _search_clique(candidates, 1)
        if pts and n > best_n:
            best_points = pts
            best_n = n
            best_D = 1
    
    # Search other D values
    if time.time() - T0 < TIME_LIMIT - 10:
        for D in [2, 3, 5, 6, 7, 10, 13, 15, 17, 19]:
            if time.time() - T0 > TIME_LIMIT - 5:
                break
            
            candidates = [(Fraction(x), Fraction(y)) for x in range(-20, 21) for y in range(-20, 21)]
            pts, n = _search_clique(candidates, D)
            if pts and n > best_n:
                best_points = pts
                best_n = n
                best_D = D
    
    # Ensure we always return at least the seed
    if best_points is None:
        best_points = seed_points
        best_D = 1
    
    return _format_result(best_points, best_D)
