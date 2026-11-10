"""Integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from itertools import combinations

def construct():
    t_start = time.time()
    DEADLINE = t_start + 145

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        return det == 0

    def validate_point_with_set(p, points):
        # Check distance to all points
        for q in points:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            if not is_square(dx*dx + dy*dy):
                return False
        
        # Check collinearity with all pairs
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], p):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], p):
                        return False
        return True

    def validate_pair(p1, p2, points):
        # Check if p1 and p2 can be added together to points
        # 1. p1 valid with points
        # 2. p2 valid with points
        # 3. p1 and p2 valid with each other and points
        if not validate_point_with_set(p1, points):
            return False
        if not validate_point_with_set(p2, points):
            return False
        
        # Check p1-p2 distance
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        if not is_square(dx*dx + dy*dy):
            return False
            
        # Check collinearity of p1, p2 with any point in points
        for p in points:
            if check_collinear(p1, p2, p):
                return False
            
        # Check concyclicity of p1, p2 with any pair in points
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_concyclic(p1, p2, points[i], points[j]):
                    return False
        return True

    # Start with a known 5-point integral set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    point_set = set(best_points)

    def find_clique_in_candidates(candidates, points):
        # candidates: list of points that are individually valid w.r.t 'points'
        # We want to find a subset of candidates that are mutually compatible.
        # Simple greedy or backtracking for small cliques.
        if not candidates:
            return []
        
        # Sort candidates by degree in the compatibility graph? 
        # For now, just try to build a clique.
        clique = []
        for c in candidates:
            # Check if c is compatible with all in current clique
            ok = True
            for q in clique:
                # Distance
                dx = c[0] - q[0]
                dy = c[1] - q[1]
                if not is_square(dx*dx + dy*dy):
                    ok = False
                    break
                # Collinear with any point in 'points'
                for p in points:
                    if check_collinear(c, q, p):
                        ok = False
                        break
                if not ok: break
                # Concyclic with any pair in 'points'
                n = len(points)
                for i in range(n):
                    for j in range(i+1, n):
                        if check_concyclic(c, q, points[i], points[j]):
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            
            if ok:
                clique.append(c)
                if len(clique) > 0: # We found at least one
                    break # Just adding one is a win. We can try to add more in next iterations.
                    # Actually, let's try to maximize clique size locally.
        
        # Try to expand the clique found
        # The above loop is greedy. Let's do a simple backtracking for small sizes.
        # Given time constraints, greedy is safer.
        return clique

    search_radius = 200
    
    while time_ok() and len(best_points) < 12:
        # Generate candidates
        candidates = []
        # Expand search radius slowly
        for x in range(-search_radius, search_radius + 1):
            if not time_ok(): break
            for y in range(-search_radius, search_radius + 1):
                if not time_ok(): break
                p = (x, y)
                if p in point_set:
                    continue
                # Quick distance check to first point to filter out most
                # (0,0) is in best_points
                dx = x
                dy = y
                if not is_square(dx*dx + dy*dy):
                    continue
                
                # Full validation
                if validate_point_with_set(p, best_points):
                    candidates.append(p)
        
        if not candidates:
            search_radius += 20
            if search_radius > 500:
                break
            continue
        
        # Try to find a clique among candidates
        # We can add points one by one, but checking compatibility with new points is key.
        # Let's try to add the "best" candidate.
        # Heuristic: Candidate that is compatible with the most other candidates?
        # Or just pick the first one?
        
        # Let's try to add a pair if possible, else single.
        added = False
        
        # Try to find a pair (p1, p2) in candidates that are mutually compatible
        # This is O(M^2) where M is num candidates.
        # If M is small (<100), this is fine.
        if len(candidates) < 100:
            for i in range(len(candidates)):
                if not time_ok(): break
                for j in range(i+1, len(candidates)):
                    if not time_ok(): break
                    p1 = candidates[i]
                    p2 = candidates[j]
                    if validate_pair(p1, p2, best_points):
                        best_points.append(p1)
                        best_points.append(p2)
                        point_set.add(p1)
                        point_set.add(p2)
                        added = True
                        break
                if added: break
        
        if not added:
            # Try to add single best candidate
            # Pick one that maximizes some score? Just pick first.
            # Maybe pick one that is "central"?
            for p in candidates:
                if not time_ok(): break
                # Re-validate just in case
                if validate_point_with_set(p, best_points):
                    best_points.append(p)
                    point_set.add(p)
                    added = True
                    break
        
        if not added:
            search_radius += 20
            if search_radius > 500:
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
