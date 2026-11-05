import random
import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0

    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        rows = [
            [a1, x1, y1, 1],
            [a2, x2, y2, 1],
            [a3, x3, y3, 1],
            [a4, x4, y4, 1]
        ]
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def check_new_point(base, new_pt):
        x, y = new_pt
        for i, (bx, by) in enumerate(base):
            dx = x - bx
            dy = y - by
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
        n = len(base)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(base[i], base[j], new_pt):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(base[i], base[j], base[k], new_pt):
                        return False
        return True

    def generate_candidates(radius):
        candidates = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x == 0 and y == 0:
                    continue
                d2 = x*x + y*y
                if is_perfect_square(d2):
                    candidates.append((x, y))
        return candidates

    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not is_valid_set(seed_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in [(0,0), (3,4), (4,3)]]}

    best_set = seed_points
    best_size = len(seed_points)
    
    search_radius = 800
    candidates = generate_candidates(search_radius)
    
    random.seed(42)
    random.shuffle(candidates)
    
    current_set = list(seed_points)
    current_set_set = set(current_set)
    
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        
        valid_cands = []
        for c in candidates:
            if c in current_set_set:
                continue
            ok = True
            for b in current_set:
                dx = c[0] - b[0]
                dy = c[1] - b[1]
                if not is_perfect_square(dx*dx + dy*dy):
                    ok = False
                    break
            if ok:
                valid_cands.append(c)
        
        if not valid_cands:
            break
            
        found = False
        for c in valid_cands:
            if check_new_point(current_set, c):
                current_set.append(c)
                current_set_set.add(c)
                found = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                break
        
        if not found:
            break

    if len(current_set) > len(best_set):
        best_set = current_set

    pts = best_set
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
