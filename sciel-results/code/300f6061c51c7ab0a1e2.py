import math
from fractions import Fraction
from itertools import combinations
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (150s limit)
    
    def elapsed():
        return time.time() - start_time
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = math.isqrt(n)
        return r * r == n
    
    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0
    
    def is_concyclic(p1, p2, p3, p4):
        """Check if 4 points are concyclic using determinant method."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Circumcircle equation: x^2 + y^2 + Dx + Ey + F = 0
        # For 4 points to be concyclic, the determinant must be 0
        # |x^2+y^2, x, y, 1|
        # |x1^2+y1^2, x1, y1, 1|
        # |x2^2+y2^2, x2, y2, 1|
        # |x3^2+y3^2, x3, y3, 1|
        # |x4^2+y4^2, x4, y4, 1|
        
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        # 4x4 determinant expansion
        # Using cofactor expansion along first row
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        # Expand along first row
        M = [[a1, x1, y1, 1],
             [a2, x2, y2, 1],
             [a3, x3, y3, 1],
             [a4, x4, y4, 1]]
        
        det = 0
        signs = [1, -1, 1, -1]
        for j in range(4):
            minor = [[M[i][k] for k in range(4) if k != j] for i in range(1, 4)]
            det += signs[j] * M[0][j] * det3(minor)
        
        return det == 0
    
    def all_distances_integer(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        return True
    
    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        return True
    
    def no_four_concyclic(points):
        n = len(points)
        for combo in combinations(range(n), 4):
            if is_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
                return False
        return True
    
    def is_valid(points):
        if not all_distances_integer(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    def try_add_point(points, new_p):
        """Try to add a new point to the set, returning new set or None."""
        if new_p in points:
            return None
        
        # Check distances to all existing points
        for p in points:
            dx = new_p[0] - p[0]
            dy = new_p[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return None
        
        # Check no three collinear
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if check_collinear(points[i], points[j], new_p):
                    return None
        
        # Check no four concyclic
        for combo in combinations(range(len(points)), 3):
            if is_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], new_p):
                return None
        
        return points + [new_p]
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    assert is_valid(base_points), "Base set invalid"
    
    best_points = list(base_points)
    best_count = 5
    
    # Strategy: Use Pythagorean triples to generate candidate points
    # For point (x, y) to have integer distance to origin, x^2 + y^2 must be a perfect square
    # Generate many Pythagorean triples and use them as candidates
    
    # Generate Pythagorean triples up to a certain size
    pyth_triples = set()
    max_val = 200
    for a in range(1, max_val):
        for b in range(a, max_val):
            c2 = a*a + b*b
            c = math.isqrt(c2)
            if c*c == c2:
                pyth_triples.add((a, b))
                pyth_triples.add((a, -b))
                pyth_triples.add((-a, b))
                pyth_triples.add((-a, -b))
                pyth_triples.add((b, a))
                pyth_triples.add((b, -a))
                pyth_triples.add((-b, a))
                pyth_triples.add((-b, -a))
    
    # Also add points that work with the existing base points
    # For each base point, find points at integer distance from it
    candidate_points = set()
    
    # Add points at integer distance from each base point
    for bp in base_points:
        for dx in range(-150, 151):
            for dy in range(-150, 151):
                d2 = dx*dx + dy*dy
                if is_perfect_square(d2):
                    candidate_points.add((bp[0] + dx, bp[1] + dy))
    
    # Also add points from Pythagorean triples relative to origin
    for px, py in pyth_triples:
        candidate_points.add((px, py))
    
    # Convert to list for iteration
    candidates = list(candidate_points)
    random.shuffle(candidates)
    
    # Try to find 6th point
    found_sixth = False
    for new_p in candidates:
        if elapsed() > deadline - 30:
            break
        
        result = try_add_point(best_points, new_p)
        if result is not None:
            best_points = result
            best_count = 6
            found_sixth = True
            break
    
    # If we found 6 points, try to find 7th
    if best_count == 6 and elapsed() < deadline - 20:
        # Generate more candidates around the 6-point set
        new_candidates = set()
        for bp in best_points:
            for dx in range(-200, 201):
                for dy in range(-200, 201):
                    d2 = dx*dx + dy*dy
                    if is_perfect_square(d2):
                        new_candidates.add((bp[0] + dx, bp[1] + dy))
        
        new_candidates = list(new_candidates)
        random.shuffle(new_candidates)
        
        found_seventh = False
        for new_p in new_candidates:
            if elapsed() > deadline - 15:
                break
            
            result = try_add_point(best_points, new_p)
            if result is not None:
                best_points = result
                best_count = 7
                found_seventh = True
                break
    
    # If we found 7 points, try to find 8th
    if best_count == 7 and elapsed() < deadline - 10:
        new_candidates = set()
        for bp in best_points:
            for dx in range(-250, 251):
                for dy in range(-250, 251):
                    d2 = dx*dx + dy*dy
                    if is_perfect_square(d2):
                        new_candidates.add((bp[0] + dx, bp[1] + dy))
        
        new_candidates = list(new_candidates)
        random.shuffle(new_candidates)
        
        found_eighth = False
        for new_p in new_candidates:
            if elapsed() > deadline - 5:
                break
            
            result = try_add_point(best_points, new_p)
            if result is not None:
                best_points = result
                best_count = 8
                found_eighth = True
                break
    
    # Final verification
    if not is_valid(best_points):
        best_points = list(base_points)
        best_count = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
